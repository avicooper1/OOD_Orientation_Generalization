import torchvision.models as models
import timm
import torch
import torch.optim as optim
import torch.nn as nn
from train import train
import os
import tqdm
import json

import sys
sys.path.append('/home/avic/Rotation-Generalization')
from dataset import RotationDataset
from my_dataclasses import ExpData, ModelType
from my_models.C8SteerableCNN import C8SteerableCNN
from my_models.CORnet_S import CORnet_S

with open('/home/avic/Rotation-Generalization/train/remaining_jobs.json') as remaining_jobs_file:
    remaining_jobs = json.load(remaining_jobs_file)
JOB_ID = remaining_jobs[int(sys.argv[-1])]

EXP_DATA = ExpData.get_experiments(JOB_ID)

dataset = RotationDataset(EXP_DATA)

if EXP_DATA.model_type == ModelType.ResNet:
    if EXP_DATA.pretrained:
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 50)
    else:
        model = models.resnet18(pretrained=False, num_classes=50)

elif EXP_DATA.model_type == ModelType.DenseNet:
    if EXP_DATA.pretrained:
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 50)
    else:
        model = models.densenet121(pretrained=False, num_classes=50)

elif EXP_DATA.model_type == ModelType.Inception:
    if EXP_DATA.pretrained:
        model = models.inception_v3(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 50)
    else:
        model = models.inception_v3(pretrained=False, num_classes=50)

elif EXP_DATA.model_type == ModelType.CorNet:
    model = CORnet_S(num_classes=50)

elif EXP_DATA.model_type == ModelType.ViT:
    model = timm.models.vit_base_patch16_224(pretrained=EXP_DATA.pretrained, num_classes=50)

elif EXP_DATA.model_type == ModelType.DeiT:
    model = timm.models.vit_deit_base_patch16_224(pretrained=EXP_DATA.pretrained, num_classes=50)

elif EXP_DATA.model_type == ModelType.Equivariant:
    model = C8SteerableCNN(n_classes=50)

EPOCH_START = 0

if os.path.exists(EXP_DATA.stats):
    with open(EXP_DATA.stats) as f:
        for i, l in enumerate(f):
            pass
    EPOCH_START = i
    if EPOCH_START >= EXP_DATA.max_epoch:
        print(f'Already trained for {EXP_DATA.max_epoch} epochs required by {EXP_DATA.model_type}. Exiting')
        exit()
    model.load_state_dict(torch.load(EXP_DATA.checkpoint))
    EXP_DATA.log(f'Starting from epoch {EPOCH_START}')

model.cuda()

EXP_DATA.log([
    EXP_DATA.__repr__(print=True),
    str(dataset),
    f'\n\n{str(model)}\n\n'])
EXP_DATA.log('Beginning training')
train(model, dataset, nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=EXP_DATA.lr), EXP_DATA, EPOCH_START, EXP_DATA.max_epoch)
EXP_DATA.log('Completed training')
