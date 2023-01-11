from functools import reduce
from torch import no_grad, squeeze, mean
from tqdm.auto import tqdm
import sys
from utils.persistent_data_class import *
from torch import save, empty
from math import ceil


def run_epoch(exp_data, model, criterion, optimizer, train_epoch, dataloader, pbar, epoch_corrects=None, num_batches=None, batch_activations=None):
    
    if train_epoch:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.
    batch_counter = 0
    correct_counter = 0
    total_counter = 0

    for images, targets in dataloader:

        targets = targets.cuda(non_blocking=True)

        # TODO: Fix this. the dataloader shouldn't be adding an extra dimension
        if len(images.shape) > 4:
            images = images[0]
            targets = targets[0]
        
        if train_epoch:
            optimizer.zero_grad()
            predictions = model(images)
        else:
            with no_grad():
                predictions = model(images)
                
        match exp_data.loss:
            case 'CE':
                current_loss = criterion(predictions, targets)
            case 'Contrastive':
                current_loss = criterion(batch_activations[0][0], predictions, targets)

        if train_epoch:
            current_loss.backward()
            optimizer.step()

        epoch_loss += current_loss

        batch_corrects = targets == predictions.argmax(dim=1)
        correct_counter = batch_corrects.sum()
        total_counter = len(targets)
        if epoch_corrects is not None:
            epoch_corrects[batch_counter][:len(targets)] = batch_corrects

        pbar.update()
        
        batch_counter += 1
        if num_batches and batch_counter > num_batches:
            break
        
    return np.round((epoch_loss / total_counter).item(), 7), np.round((correct_counter / total_counter).item(), 7)


def train(model, dataset, criterion, optimizer, exp_data):

    activations = [empty(ceil(len(loader.dataset.frame) / exp_data.batch_size), exp_data.batch_size).cuda() for loader in dataset.data_loaders[1:]]
    corrects = [empty(ceil(len(loader.dataset.frame) / exp_data.batch_size), exp_data.batch_size).cuda() for loader in dataset.data_loaders[1:]]
    loader_index = None
    activations_counter = 0
    corrects_index = None
    batch_activations = [None]

    penultimate_layer = reduce(getattr, [model] + exp_data.hook_layer.split('.'))
    
    def store_activations(m, i, o):
        if loader_index:
            activations[loader_index][activations_counter][:i.shape[0]] = i
            activations_counter += 1
        if exp_data.loss == 'Contrastive':
            batch_activations[0] = i

    penultimate_layer.register_forward_hook(store_activations)

    num_training_batches = 1000
    total_num_batches = num_training_batches + sum([len(l) for l in dataset.data_loaders[1:]])

    for epoch in tqdm(range(exp_data.epochs_completed, exp_data.max_epochs),
                      desc='Training epochs completed',
                      file=sys.stdout):
        with tqdm(total=total_num_batches, leave=False, file=sys.stdout) as pbar:

            loader_index = None

            training_loss, training_accuracy = run_epoch(exp_data,
                                      model,
                                      criterion,
                                      optimizer,
                                      True,
                                      dataset.train_loader,
                                      pbar,
                                      num_batches=num_training_batches,
                                      batch_activations=batch_activations)
            
            exp_data.eval_data.training_losses.append(training_loss)
            exp_data.eval_data.training_accuracies.append(np.mean(training_accuracy))

            for loader_index, (dataloader,
                               accuracies,
                               losses) in enumerate(zip(dataset.eval_loaders,
                                                             exp_data.eval_data.accuracies,
                                                             exp_data.eval_data.losses)):
                
                activations_counter = 0
                loader_corrects = corrects[loader_index]
                loss, accuracy = run_epoch(exp_data, model, criterion, None, False, dataloader, pbar, epoch_corrects=loader_corrects, batch_activations=batch_activations)
                
                losses.append(loss)
                accuracies.append(accuracy)
        
        exp_data.eval_data.validation_and_partial_base_accuracies.append(((exp_data.eval_data.full_validation_accuracies[-1] * len(exp_data.full_instances)) + exp_data.eval_data.partial_base_accuracies[-1] * len(exp_data.partial_instances)) / (len(exp_data.full_instances) + len(exp_data.partial_instances)))
        exp_data.eval_data.save()
        
        max_accuracy_index = np.argmax(exp_data.eval_data.validation_and_partial_base_accuracies)
        
        if max_accuracy_index == len(exp_data.eval_data.validation_and_partial_base_accuracies) - 1:
            for i, (loader_corrects, loader_activations) in enumerate(zip(exp_data.eval_data.corrects,
                                                            exp_data.eval_data.activations)):
                loader_corrects.arr = corrects[i].flatten().detach().cpu().numpy()
                loader_activations.arr = activations[i].flatten().detach().cpu().numpy()

        save({'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()},
             exp_data.checkpoint)

        exp_data.epochs_completed = epoch + 1
        exp_data.save()

        num_epochs_after_peak = 7
        if (exp_data.epochs_completed >= (exp_data.min_epochs + num_epochs_after_peak)) and (max_accuracy_index < (len(exp_data.eval_data.validation_and_partial_base_accuracies) - num_epochs_after_peak)):
            break

    for corrects, activations in zip(exp_data.eval_data.corrects, exp_data.eval_data.activations):
        
        corrects.dump()
        activations.dump()
