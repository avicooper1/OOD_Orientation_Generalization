from torch.backends.cuda import matmul
import timm
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import os
import sys
import argparse
from contextlib import redirect_stdout
from contextlib import nullcontext

if __name__ == '__main__':
    
    matmul.allow_tf32 = True

    parser = argparse.ArgumentParser(description='Run an experiment')
    parser.add_argument('project_path', help='path to the project source directory')
    parser.add_argument('storage_path', help='path to the storage directory')
    parser.add_argument('job_id', type=int, help="slurm job id")
    parser.add_argument('log_to_exp', type=bool, nargs='?', const=True, default=False)

    args = parser.parse_args()

    sys.path.insert(0, args.project_path)

    from train import train
    from utils.dataset import RotationDataset
    from utils.persistent_data_class import ExpData

    EXP_DATA = ExpData.from_job_i(args.project_path, args.storage_path, args.job_id, create_exp=True)

    with open(EXP_DATA.log, 'a') as out:
        with redirect_stdout(out):
    
            dataset = RotationDataset(EXP_DATA)

            match len(timm.list_models(EXP_DATA.model_type)):
                case 0:
                    # TODO: implement other models not in timm
                    pass
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
            train(model, dataset, CrossEntropyLoss(), Adam(model.parameters(), lr=EXP_DATA.lr), EXP_DATA)
            print('Completed Training')
