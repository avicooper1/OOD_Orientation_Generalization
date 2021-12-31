import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import sys
sys.path.append('/home/avic/Rotation-Generalization')
from my_dataclasses import *

def train(model, dataset, criterion, optimizer, exp_data: ExpData, epoch_start, epoch_end, topk=False):
    def run_epoch(train: bool, split: DataLoader, tqdm_disable=True):

        if train:
            model.train()
        else:
            model.eval()

        epoch_loss = 0.
        all_preds = []
        all_targets = []

        for batch in tqdm(split, disable=tqdm_disable):
            if train:
                optimizer.zero_grad()
            images, targets = map(lambda t: t.to(device='cuda', non_blocking=True), batch)
            if train:
                preds = model(images)
            else:
                with torch.no_grad():
                    preds = model(images)
            targets = targets.flatten()

            if exp_data.model_type == ModelType.Inception and train:
                current_loss = criterion(preds.logits, targets) + criterion(preds.aux_logits, targets)
            else:
                current_loss = criterion(preds, targets)

            if train:
                current_loss.backward()
                optimizer.step()

            epoch_loss += current_loss
            # if not topk:
            all_preds.append((preds.logits if (exp_data.model_type == ModelType.Inception) and train else preds).argmax(dim=1).cpu())
            # else:
            #     all_preds.append(model(images).topk(topk)[1].cpu())
            all_targets.append(targets.cpu())
        return epoch_loss / len(split.dataset), (np.concatenate(all_targets), np.concatenate(all_preds))

    writer = SummaryWriter(log_dir=exp_data.tensorboard_logs_dir)

    def get_accuracy(targets, preds):
        if not topk:
            return preds == targets
        else:
            print("Error: topk not implemented")
            # truth_matrix = np.empty_like(targets, dtype=np.bool)
            # for i, (t,p) in enumerate(zip(targets, preds)):
            #     truth_matrix[i] = np.isin(t,p)

    for epoch in tqdm(range(epoch_start, epoch_end), disable=False):

        stats = {'Epoch': epoch}

        for split, name in dataset.splits.all_loaders_named:
            dataset.set_augmentation(exp_data.augment and name == 'Training')
            results = run_epoch(name == 'Training', split)
            accuracy = get_accuracy(results[1][0], results[1][1])

            loss_string = f'{name} Loss'
            accuracy_string = f'{name} Accuracy'

            writer.add_scalar(loss_string, results[0], epoch)
            writer.add_scalar(accuracy_string, np.average(accuracy), epoch)

            stats[loss_string] = round(results[0].item(), 7)
            stats[accuracy_string] = np.average(accuracy)

            if name == 'Testing':

                pd.DataFrame({
                    'data_div': exp_data.data_div,
                    'epoch': epoch,
                    'image_name': dataset.splits.testing.dataset.frame.image_name,
                    'predicted_model': dataset.obj_cat_codec.inverse_transform(results[1][1]),
                    'correct': get_accuracy(results[1][0], results[1][1])}).to_csv(exp_data.eval, mode='a', header=not os.path.exists(exp_data.eval))
        pd.DataFrame(stats, index=[0]).to_csv(exp_data.stats, mode='a', header=not os.path.exists(exp_data.stats))
        torch.save(model.state_dict(), exp_data.checkpoint)

        exp_data.log([f'Finished epoch {epoch}'])
