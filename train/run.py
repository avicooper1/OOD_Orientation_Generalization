from torch.backends.cuda import matmul
from torch.nn import Linear, Conv2d
import timm
from cornet import CORnet_S
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import sys
import argparse
from contextlib import redirect_stdout

import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feature_vectors, labels):
        # Normalize feature vectors
        feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)
        # Compute logits
        logits = torch.div(
            torch.matmul(
                feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
            ),
            self.temperature,
        )
        return losses.NTXentLoss(temperature=0.07)(logits, torch.squeeze(labels))

if __name__ == '__main__':
    
    matmul.allow_tf32 = True

    parser = argparse.ArgumentParser(description='Run an experiment')
    parser.add_argument('project_path', help='path to the project source directory')
    parser.add_argument('storage_path', help='path to the storage directory')
    parser.add_argument('job_id', type=int, help="slurm job id")

    args = parser.parse_args()

    sys.path.insert(0, args.project_path)

    from SupContrast.losses import SupConLoss

    from train import train
    from utils.dataset import RotationDataset
    from utils.persistent_data_class import ExpData

    EXP_DATA = ExpData.from_job_i(args.project_path, args.storage_path, args.job_id, create_exp=True)

    # with open(EXP_DATA.log, 'a') as out:
    #     with redirect_stdout(out):

    dataset = RotationDataset(EXP_DATA)

    match len(timm.list_models(EXP_DATA.model_type)):
        case 0:
            match EXP_DATA.model_type:
                case 'cornet':
                    model = CORnet_S()
                    model.V1.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                    model.decoder.linear = Linear(in_features=512, out_features=50, bias=True)
        case 1:
            model = timm.create_model(EXP_DATA.model_type, in_chans=1,
                                      pretrained=EXP_DATA.pretrained,
                                      num_classes=50)
        case _:
            print(f'Multiple timm models match model_type: {EXP_DATA.model_type}. Model choice is ambiguous. Exiting.')
            exit(-1)

    assert EXP_DATA.epochs_completed == 0, f'Already trained for {EXP_DATA.epochs_completed} epochs. We currently do not support reloading models during training.'

    model.cuda()

    print('Beginning Training')
    # train(model, dataset, CrossEntropyLoss(), Adam(model.parameters(), lr=EXP_DATA.lr), EXP_DATA)
    train(model, dataset, SupConLoss().cuda(), Adam(model.parameters(), lr=0.05), EXP_DATA)
    # train(model, dataset, SupervisedContrastiveLoss(), Adam(model.parameters(), lr=0.05), EXP_DATA)
    print('Completed Training')
