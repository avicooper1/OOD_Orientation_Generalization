from torch import load
from torch.backends.cuda import matmul
from torchvision.models import resnet18, densenet121, inception_v3
# import timm
from torch.optim import Adam
from torch.nn import Linear, CrossEntropyLoss
import os
import sys
import argparse
from contextlib import redirect_stdout

#with open('/home/avic/Rotation-Generalization/train/remaining_jobs.json') as remaining_jobs_file:
#    remaining_jobs = json.load(remaining_jobs_file)
#JOB_ID = remaining_jobs[int(sys.argv[-1])]

if __name__ == '__main__':
    
    matmul.allow_tf32 = True

    parser = argparse.ArgumentParser(description='Run an experiment')
    parser.add_argument('project_path', help='path to the project source directory')
    parser.add_argument('storage_path', help='path to the storage directory')
    parser.add_argument('job_id', type=int, help="slurm job id")

    args = parser.parse_args()

    sys.path.insert(0, args.project_path)

    from train import train
    from utils.dataset import RotationDataset
    from utils.persistent_data_class import ExpData, ModelType

    EXP_DATA = ExpData.from_job_i(args.project_path, args.storage_path, args.job_id)
    
    # with open(EXP_DATA.log, 'ab', 128) as f:
    #     with redirect_stdout(f):

    dataset = RotationDataset(EXP_DATA)

    if os.path.exists(EXP_DATA.checkpoint):
        model = load(EXP_DATA.checkpoint)
    else:
        if EXP_DATA.model_type == ModelType.ResNet:
            if EXP_DATA.pretrained:
                model = resnet18(pretrained=True)
                model.fc = Linear(model.fc.in_features, 50)
            else:
                model = resnet18(weights=None, num_classes=50)

        elif EXP_DATA.model_type == ModelType.DenseNet:
            if EXP_DATA.pretrained:
                model = densenet121(pretrained=True)
                model.classifier = Linear(model.classifier.in_features, 50)
            else:
                model = densenet121(weights=None, num_classes=50)

        elif EXP_DATA.model_type == ModelType.Inception:
            if EXP_DATA.pretrained:
                model = inception_v3(pretrained=True)
                model.fc = Linear(model.fc.in_features, 50)
            else:
                model = inception_v3(pretrained=False, num_classes=50)

        elif EXP_DATA.model_type == ModelType.CorNet:
            print('CorNet not implemented')
            exit()
            model = CORnet_S(num_classes=50)

        elif EXP_DATA.model_type == ModelType.ViT:
            model = timm.models.vit_base_patch16_224(pretrained=EXP_DATA.pretrained, num_classes=50)

        elif EXP_DATA.model_type == ModelType.DeiT:
            model = timm.models.vit_deit_base_patch16_224(pretrained=EXP_DATA.pretrained, num_classes=50)

    if EXP_DATA.epochs_completed >= EXP_DATA.max_epochs:
        print(f'Already trained for {EXP_DATA.max_epoch} epochs required by {EXP_DATA.model_type}. Exiting')
        exit()
    else:
        print(f'Continuing training, starting from epoch {EXP_DATA.epochs_completed}')

    model.cuda()

    print('Beginning Training')
    train(model, dataset, CrossEntropyLoss(), Adam(model.parameters(), lr=EXP_DATA.lr), EXP_DATA)
    print('Completed Training')
