from torch import no_grad, squeeze
from tqdm.auto import tqdm
import numpy as np
import sys
from utils.persistent_data_class import *


def run_epoch(exp_data, model, criterion, optimizer, train_epoch, dataloader, pbar, num_batches=0):
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

        if len(images.shape) > 4:
            images = images[0]
            targets = targets[0]
        
        if train_epoch:
            optimizer.zero_grad()
            predictions = model(images)
        else:
            with no_grad():
                predictions = model(images)

        if exp_data.model_type == ModelType.Inception:
            current_loss = criterion(predictions.logits, targets) + criterion(predictions.aux_logits, targets)
        else:
            current_loss = criterion(predictions, targets)

        if train_epoch:
            current_loss.backward()
            optimizer.step()

        epoch_loss += current_loss

        all_predictions.append(
            (predictions.logits.cpu().numpy() if (exp_data.model_type == ModelType.Inception) and train_epoch else predictions).argmax(dim=1).cpu().numpy())
        all_targets.append(targets.cpu().numpy())

        pbar.update()
        
        batch_counter += 1
        if num_batches != 0 and batch_counter > num_batches:
            break
        
    return np.round((epoch_loss / len(dataloader.dataset)).item(), 7), np.concatenate(all_targets) == np.concatenate(all_predictions)


def train(model, dataset, criterion, optimizer, exp_data):

    epoch_activations = []
    def hook(model, input, output):
        epoch_activations.append(squeeze(output).detach().cpu().numpy())

    model.avgpool.register_forward_hook(hook)
    
    num_training_batches = 1000
    total_num_batches = num_training_batches + sum([len(l) for l in dataset.data_loaders[1:]])  
    
    peak_partial_ood_subset_accuracy = None

    for epoch in tqdm(range(exp_data.epochs_completed, exp_data.max_epochs), desc='Training epochs completed', file=sys.stdout):
        with tqdm(total=total_num_batches, leave=False, file=sys.stdout) as pbar:
            
            training_loss, training_accuracy = run_epoch(exp_data, model, criterion, optimizer, True, dataset.train_loader, pbar, num_training_batches)
            exp_data.eval_data.training_losses.append(training_loss)
            exp_data.eval_data.training_accuracies.append(np.mean(training_accuracy))

            # Determine if network is at peak accuracy for partial OOD by evaluating on a subset of the frame. If at peak, save corrects and run analysis
            # on full partial OOD frame
            partial_ood_subset_correct = run_epoch(exp_data, model, criterion, None, False, dataset.partial_ood_subset_loader, pbar)[1]
            partial_ood_subset_accuracy = np.round(np.mean(partial_ood_subset_correct), 7)
            if peak_partial_ood_subset_accuracy is None:
                peak_partial_ood_subset_accuracy = (True, partial_ood_subset_accuracy)
            else:
                if peak_partial_ood_subset_accuracy[1] < partial_ood_subset_accuracy:
                    peak_partial_ood_subset_accuracy = (True, partial_ood_subset_accuracy)
                else:
                    peak_partial_ood_subset_accuracy = (False, peak_partial_ood_subset_accuracy[1])

            for dataloader, corrects, accuracies, losses, activations in zip(dataset.eval_loaders,
                                                                             exp_data.eval_data.corrects,
                                                                             exp_data.eval_data.accuracies,
                                                                             exp_data.eval_data.losses,
                                                                             exp_data.eval_data.activations):

                if dataloader.dataset.name == 'partial OOD' and not peak_partial_ood_subset_accuracy[0]:
                    losses.append(None)
                    accuracies.append(None)
                    continue
                
                epoch_activations = []
                loss, corrects.arr = run_epoch(exp_data, model, criterion, None, False, dataloader, pbar)
                
                accuracy = np.round(np.mean(corrects.arr), 7)
                
                if peak_partial_ood_subset_accuracy[0]:
                    corrects.dump()
                    activations.arr = np.vstack(epoch_activations)
                    activations.dump()
                    
                losses.append(loss)
                accuracies.append(accuracy)
            
        if exp_data.eval_data.partial_ood_accuracies[-5:] == [None] * 5:
            break

        exp_data.eval_data.save()
        # save(model.state_dict(), exp_data.checkpoint)
