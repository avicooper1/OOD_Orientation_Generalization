from functools import reduce
from torch import no_grad, squeeze
from tqdm.auto import tqdm
import numpy as np
import sys
from utils.persistent_data_class import *
from torch import broadcast_to, unsqueeze
from torch.nn import BatchNorm2d
from torch.nn.functional import normalize
import torch
from torch.nn import CrossEntropyLoss


def run_epoch(exp_data, model, criterion, optimizer, train_epoch, dataloader, pbar, num_batches=0, batch_activations=None):
    
    if train_epoch:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.
    all_predictions = []
    all_targets = []
    batch_counter = 0

    for images, targets in dataloader:

        targets = targets.cuda(non_blocking=True)

        # TODO: Fix this. ie the dataloader shouldn't be adding an extra dimension
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

        all_targets.append(targets.cpu().numpy())
        all_predictions.append(predictions.argmax(dim=1).cpu().numpy())

        pbar.update()
        
        batch_counter += 1
        if num_batches != 0 and batch_counter > num_batches:
            break
        
    return np.round((epoch_loss / len(dataloader.dataset)).item(), 7), np.concatenate(all_targets) == np.concatenate(all_predictions)


def train(model, dataset, criterion, optimizer, exp_data):

    epoch_activations = []
    store_epoch_activations = False
    batch_activations = [None]

    penultimate_layer = reduce(getattr, [model] + exp_data.hook_layer.split('.'))
    
    def store_activations(m, i, o):
        if store_epoch_activations:
            epoch_activations.append(squeeze(i[0]).detach().cpu().numpy())
        if exp_data.loss == 'Contrastive':
            batch_activations[0] = i
        
    
    penultimate_layer.register_forward_hook(store_activations)
    
    
    num_training_batches = 1000
    total_num_batches = num_training_batches + sum([len(l) for l in dataset.data_loaders[1:]])

    for exp_data.epochs_completed in tqdm(range(exp_data.epochs_completed, exp_data.max_epochs),
                      desc='Training epochs completed',
                      file=sys.stdout):
        with tqdm(total=total_num_batches, leave=False, file=sys.stdout) as pbar:
            
            store_epoch_activations = False

            training_loss, training_accuracy = run_epoch(exp_data,
                                                         model,
                                                         criterion,
                                                         optimizer,
                                                         True,
                                                         dataset.train_loader,
                                                         pbar,
                                                         num_training_batches,
                                                         batch_activations)
            
            exp_data.eval_data.training_losses.append(training_loss)
            exp_data.eval_data.training_accuracies.append(np.mean(training_accuracy))
            
            # peak_partial_ood_accuracy = False
            
            store_epoch_activations = True
            
            # The list comprehension in the following line allows us to reorder the data loaders and arrays so that
            # the partial base set is shown first, which allows us to evaluate if network is at peak performance
            for dataloader, corrects, accuracies, losses, activations in list(zip(dataset.eval_loaders,
                                                                                  exp_data.eval_data.corrects,
                                                                                  exp_data.eval_data.accuracies,
                                                                                  exp_data.eval_data.losses,
                                                                                  exp_data.eval_data.activations)):
                
                epoch_activations = []
                loss, epoch_corrects = run_epoch(exp_data, model, criterion, None, False, dataloader, pbar, batch_activations=batch_activations)
                
                accuracy = np.round(np.mean(epoch_corrects), 7)
                
                losses.append(loss)
                accuracies.append(accuracy)
                
                corrects.temp_arr = epoch_corrects
                activations.temp_arr = np.vstack(epoch_activations)
                
#                 # The full validation loader is first in eval loaders. Thus the results of all other (subsequent)
#                 # dataloaders will be saved only if full validation is at peak accuracy
#                 if (dataloader.dataset.name == 'full validation'):
#                     num_partial_ood_accuracies = len(exp_data.eval_data.partial_ood_accuracies)
#                     if (num_partial_ood_accuracies > 0) and (np.argmax(exp_data.eval_data.partial_ood_accuracies) == num_partial_ood_accuracies - 1):
#                         peak_partial_ood_accuracy = True
                
#                 if peak_partial_ood_accuracy:
#                     corrects.arr = epoch_corrects
#                     activations.arr = np.vstack(epoch_activations)
        
        exp_data.eval_data.validation_and_partial_base_accuracies.append(((exp_data.eval_data.full_validation_accuracies[-1] * len(exp_data.full_instances)) + exp_data.eval_data.partial_base_accuracies[-1] * len(exp_data.partial_instances)) / (len(exp_data.full_instances) + len(exp_data.partial_instances)))
        exp_data.eval_data.save()
        exp_data.save()
        
        max_validation_and_partial_base_accuracies_index = np.argmax(exp_data.eval_data.validation_and_partial_base_accuracies)
        
        if max_validation_and_partial_base_accuracies_index == len(exp_data.eval_data.validation_and_partial_base_accuracies) - 1:
            for corrects, activations in zip(exp_data.eval_data.corrects, exp_data.eval_data.activations):
                corrects.arr = corrects.temp_arr
                activations.arr = activations.temp_arr
            
        
        if (exp_data.epochs_completed >= 10) and (max_validation_and_partial_base_accuracies_index < (len(exp_data.eval_data.validation_and_partial_base_accuracies) - 7)):
            break
            
    for corrects, activations in zip(exp_data.eval_data.corrects, exp_data.eval_data.activations):
        
        corrects.dump()
        activations.dump()