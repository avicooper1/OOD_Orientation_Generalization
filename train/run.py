from torch.backends.cuda import matmul
from torch.nn import Linear, Conv2d
import timm
from cornet import CORnet_S
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import sys
import argparse
from contextlib import redirect_stdout
from torch import where, randint, square, concat
from torch.nn import Module


class MyContrastiveLoss(Module):
    def __init__(self, cross_entropy_loss, temp=0.1):
        super().__init__()
        self.cross_entropy_loss = cross_entropy_loss
        self.temp = temp


    def pair_distance(self, index, label, class_map, features):
        pairs = class_map[label]
        if len(pairs) == 1:
            return 0
        return square(features[index] - features[where(pairs != label)[0][randint(0, len(pairs) - 1, (1,))[0]]]).sum().sqrt()


    def forward(self, pre_projection_activations, post_projection_activations, labels):
        class_map = [where(labels == x)[0] for x in range(50)]
        mean_distance = concat([self.pair_distance(index,
                                                   label,
                                                   class_map,
                                                   pre_projection_activations) for index, label in enumerate(labels)]).mean()
        return self.cross_entropy_loss(post_projection_activations, labels) + (self.temp * mean_distance)




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
    train(model, dataset, MyContrastiveLoss(CrossEntropyLoss()), Adam(model.parameters(), lr=0.05), EXP_DATA)
    # train(model, dataset, SupConLoss().cuda(), Adam(model.parameters(), lr=0.05), EXP_DATA)
    # train(model, dataset, SupervisedContrastiveLoss(), Adam(model.parameters(), lr=0.05), EXP_DATA)
    print('Completed Training')
