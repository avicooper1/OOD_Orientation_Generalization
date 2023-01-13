from torch.backends.cuda import matmul
from torch.nn import Linear, Conv2d
import timm
from cornet import CORnet_S
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import CrossEntropyLoss
import sys
import argparse
from contextlib import redirect_stdout
from losses import MyContrastiveLoss
from torch import load
from torch.autograd.profiler import profile


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
	from utils.persistent_data_class import ExpData

	import torch.multiprocessing as mp
	mp.set_start_method('spawn')

	EXP_DATA = ExpData.from_job_i(args.project_path, args.storage_path, args.job_id, create_exp=True)

	# with open(EXP_DATA.log, 'a') as out:
	#     with redirect_stdout(out):

	dataset = RotationDataset(EXP_DATA)

	match len(timm.list_models(EXP_DATA.model_type)):
		case 0:
			match EXP_DATA.model_type:
				case 'cornet':
					model = CORnet_S()
					model.V1.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
											padding=(3, 3), bias=False)
					model.decoder.linear = Linear(in_features=512, out_features=50, bias=True)
		case 1:
			model = timm.create_model(EXP_DATA.model_type, in_chans=1,
									  pretrained=EXP_DATA.pretrained,
									  num_classes=50)
		case _:
			print(f'Multiple timm models match model_type: {EXP_DATA.model_type}. Model choice is ambiguous. Exiting.')
			exit(-1)

	model.cuda()

	optimizer = Adam(model.parameters(), lr=0.01)
	scheduler = None #ExponentialLR(optimizer, gamma=0.98)

	if EXP_DATA.complete:
		print("Training has already completed. Exiting")
		exit()

	if EXP_DATA.epochs_completed > 0:
		checkpoint = load(EXP_DATA.checkpoint)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # I believe this is a no-op. checkpointing only
		# happens after every epoch, and optimizer is zeroed at the beginning of each epoch

		del checkpoint

	match EXP_DATA.loss:
		case 'CE':
			loss = CrossEntropyLoss()
		case 'Contrastive':
			loss = MyContrastiveLoss()
	with profile(use_cuda=True) as prof:
		print('Beginning Training')
		train(model, dataset, loss, optimizer, scheduler, EXP_DATA)
		EXP_DATA.complete = True
		EXP_DATA.save()
		print('Completed Training')
	print(prof.key_averages().table(sort_by="self_cpu_time_total"))
