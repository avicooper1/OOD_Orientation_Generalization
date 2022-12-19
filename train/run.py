from torch.backends.cuda import matmul
from torch.nn import Linear, Conv2d
import timm
from cornet import CORnet_S
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import sys
import argparse
from contextlib import redirect_stdout
from torch import where, randint, square, stack, tensor, where
from torch.nn import Module


class MyContrastiveLoss(Module):
    def __init__(self, temp=0.01):
        super().__init__()
        self.cross_entropy_loss = CrossEntropyLoss()
        self.temp = temp


    def pair_distance(self, index, label, class_map):
        pairs = class_map[label]
        if len(pairs) == 1:
            return tensor(index).cuda()
        return pairs[where(pairs != index)[0][randint(0, len(pairs) - 1, (1,))[0]]]


    def forward(self, pre_projection_activations, post_projection_activations, labels):
        class_map = [where(labels == x)[0] for x in range(50)]
        pair_indexes = stack([self.pair_distance(index, label, class_map) for index, label in enumerate(labels)])
        sum_square = square(pre_projection_activations - pre_projection_activations[pair_indexes]).sum(dim=1)
        distance = where(sum_square == 0, 0.00001, sum_square).sqrt().sum()
        return self.cross_entropy_loss(post_projection_activations, labels) + (self.temp * distance)




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

    with open(EXP_DATA.log, 'a') as out:
        with redirect_stdout(out):

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

            match EXP_DATA.loss:
                case 'CE':
                    loss = CrossEntropyLoss()
                case 'Contrastive':
                    loss = MyContrastiveLoss()

            print('Beginning Training')
            train(model, dataset, loss, Adam(model.parameters(), lr=0.01), EXP_DATA)
            print('Completed Training')
