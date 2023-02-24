from torch.backends.cuda import matmul
from torch.backends.cudnn import benchmark
from torch.nn import Linear, Conv2d
import timm
from cornet import CORnet_S
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import CrossEntropyLoss
import sys
import argparse
from contextlib import nullcontext, redirect_stdout
from losses import MyContrastiveLoss
from torch import save, load
from dataclasses import dataclass


class ModelObjects:

	def __init__(self, exp, model, optimizer, scheduler):
		self.exp = exp
		self.model = model
		self.optimizer = optimizer
		self.scheduler = scheduler

		self.model_key = 'model_state_dict'
		self.optimizer_key = 'optimizer_state_dict'
		self.scheduler_key = 'scheduler_state_dict'

	def save(self, peak):
		save({self.model_key: self.model.state_dict(),
			  self.optimizer_key: self.optimizer.state_dict(),
			  self.scheduler_key: self.scheduler.state_dict()},
			 self.exp.peak_checkpoint if peak else self.exp.checkpoint)

	def load(self, peak):
		checkpoint = load(self.exp.peak_checkpoint if peak else self.exp.checkpoint)
		self.model.load_state_dict(checkpoint[self.model_key])
		self.optimizer.load_state_dict(
			checkpoint[self.optimizer_key])  # I believe this is a no-op. checkpointing only
		# happens after every epoch, and optimizer is zeroed at the beginning of each epoch
		self.scheduler.load_state_dict(checkpoint[self.scheduler_key])

		del checkpoint


if __name__ == '__main__':

	matmul.allow_tf32 = True
	benchmark = True

	parser = argparse.ArgumentParser(description='Run an experiment')
	parser.add_argument('project_path', help='path to the project source directory')
	parser.add_argument('storage_path', help='path to the storage directory')
	parser.add_argument('job_id', type=int, help="slurm job id")
	parser.add_argument('--log_to_console', default=False, action='store_true', help='log to file, or to stdout')

	args = parser.parse_args()

	sys.path.insert(0, args.project_path)

	from train import train, evaluate
	from utils.dataset import RotationDataset
	from utils.persistent_data_class import ExpData

	COLLECT_FOR_DEEPHYS = False

	EXP_DATA = ExpData.from_job_i(args.project_path, args.storage_path, args.job_id, create_exp=True)

	with open(EXP_DATA.log, 'a') if not args.log_to_console else nullcontext(sys.stdout) as out:
		with redirect_stdout(out):

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
					print(
						f'Multiple timm models match model_type: {EXP_DATA.model_type}. Model choice is ambiguous. Exiting.')
					exit(-1)

			model.cuda()

			optimizer = Adam(model.parameters(), lr=0.01) #SGD(model.parameters(), lr=0.01)
			scheduler = ExponentialLR(optimizer, gamma=0.98)

			model_objects = ModelObjects(EXP_DATA, model, optimizer, scheduler)

			train_or_evaluate = 'train'
			if EXP_DATA.complete:
				if not EXP_DATA.eval_data.full_validation_activations.on_disk() or COLLECT_FOR_DEEPHYS:
					print("Training has already completed, but activations were not saved. Collecting activations")
					train_or_evaluate = 'evaluate'
				else:
					print("Training has already completed. Exiting")
					exit()

			if EXP_DATA.epochs_completed > 0:
				model_objects.load(train_or_evaluate == 'evaluate')

			match EXP_DATA.loss:
				case 'CE':
					loss = CrossEntropyLoss()
				case 'Contrastive':
					loss = MyContrastiveLoss()

			print('Beginning Training')
			if train_or_evaluate == 'train':
				train(model_objects, dataset, loss, EXP_DATA)
			else:
				evaluate(model_objects, dataset, loss, EXP_DATA, COLLECT_FOR_DEEPHYS)
			EXP_DATA.complete = True
			EXP_DATA.save()
			print('Completed Training')
