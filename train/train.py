from functools import reduce
from torch import no_grad, squeeze, mean, stack
from tqdm.auto import tqdm
import sys
from utils.persistent_data_class import *
from torch import save, empty
import numpy as np
from torchvision.transforms import Resize
import os


class BatchedTensorStorage:
	def __init__(self, num_batches, batch_size, trailing_dims=(), lazy=False):
		self.tensor = [empty((batch_size, *trailing_dims)) if not lazy else None for _ in range(num_batches)]
		self.num_batches = num_batches
		self.batch_size = batch_size
		self.trailing_dims = trailing_dims
		self.batch_index = -1
		self.batch_length = 0
		self.lazy = lazy

	def append(self, d):
		self.batch_length = len(d)
		self.batch_index += 1
		if not self.lazy:
			self.tensor[self.batch_index][:len(d)] = d
		else:
			self.tensor[self.batch_index] = d

	def batch(self):
		assert self.batch_index != -1
		return self.tensor[self.batch_index][:self.batch_length]

	def reset(self):
		self.batch_index = -1

	def all_data(self):
		assert self.batch_index != -1
		return stack(self.tensor).reshape((self.num_batches * self.batch_size, *self.trailing_dims))[:(self.batch_index * self.batch_size) + self.batch_length]


def run_epoch(model_objects, dataloader, loss, criterion, train_epoch, pbar, epoch_corrects=None,
			  num_batches=None, batch_activations=None, collect_for_deephys=False):

	model, optimizer, scheduler = model_objects.model, model_objects.optimizer, model_objects.scheduler

	if train_epoch:
		model.train()
	else:
		model.eval()

	epoch_loss = 0.
	batch_counter = 0
	correct_counter = 0
	total_counter = 0

	if collect_for_deephys:
		collected_images = []
		collected_targets = []
		collected_predictions = []

	for images, targets in dataloader:

		# TODO: Fix this. the dataloader shouldn't be adding an extra dimension
		if len(images.shape) > 4:
			images = images[0]
			targets = targets[0]

		if train_epoch:
			optimizer.zero_grad()
			predictions = model(images)
		else:
			with no_grad():
				predictions = model(images)

		match loss:
			case 'CE':
				current_loss = criterion(predictions, targets)
			case 'Contrastive':
				current_loss = criterion(batch_activations.batch(), predictions, targets)

		if train_epoch:
			current_loss.backward()
			optimizer.step()

		epoch_loss += current_loss

		if collect_for_deephys:
			collected_images.append(Resize((64, 64))(images)[:, 0].cpu().numpy())
			collected_targets.append(targets.cpu().numpy())
			collected_predictions.append(predictions.cpu().numpy())

		batch_corrects = targets == predictions.argmax(dim=1)
		correct_counter += batch_corrects.sum()
		total_counter += len(targets)
		if epoch_corrects is not None:
			epoch_corrects.append(batch_corrects)

		pbar.update()

		batch_counter += 1
		if num_batches and batch_counter > num_batches:
			break

	if scheduler is not None:
		scheduler.step()

	if len(dataloader.dataset) > 0:
		if collect_for_deephys:
			return np.round((epoch_loss / total_counter).item(), 7), np.round((correct_counter / total_counter).item(), 7), np.vstack(collected_images), np.concatenate(collected_targets), np.vstack(collected_predictions)
		return np.round((epoch_loss / total_counter).item(), 7), np.round((correct_counter / total_counter).item(), 7)
	else:
		return 0, 0


def run_evalutation(model_objects, dataset, criterion, exp_data, pbar, epoch_corrects, epoch_activations, dataloader_index, collect_for_deephys=False):
	for dataloader_idx, (dataloader, accuracies, losses) in enumerate(zip(dataset.eval_loaders,
																			exp_data.eval_data.accuracies,
																			exp_data.eval_data.losses), 1):

		dataloader_index[0] = dataloader_idx
		results = run_epoch(model_objects, dataloader, exp_data.loss, criterion, False, pbar,
							epoch_corrects=epoch_corrects[dataloader_index[0]],
							batch_activations=epoch_activations[dataloader_index[0]] if exp_data.loss == "Contrastive" else None,
							collect_for_deephys=collect_for_deephys)
		if collect_for_deephys:
			loss, accuracy, images, targets, predictions = results
			print(dataloader.dataset.name)

		else:
			loss, accuracy = results


		losses.append(loss)
		accuracies.append(accuracy)

		if collect_for_deephys:
			dirpath = f'/home/avic/om2/deephys/exp{exp_data.num}/'
			os.makedirs(dirpath, exist_ok=True)
			a = epoch_activations[dataloader_index[0]].all_data().cpu().numpy()
			print(images.shape, targets.shape, predictions.shape, a.shape)
			np.save(dirpath + '_'.join(dataloader.dataset.name.split(' ')) + '_activations', a)
			np.save(dirpath + '_'.join(dataloader.dataset.name.split(' ')) + '_images', images)
			np.save(dirpath + '_'.join(dataloader.dataset.name.split(' ')) + '_targets', targets)
			np.save(dirpath + '_'.join(dataloader.dataset.name.split(' ')) + '_predictions', predictions)



def setup_activation_collection(model_objects, dataset, exp_data):
	penultimate_layer = reduce(getattr, [model_objects.model] + exp_data.hook_layer.split('.'))

	epoch_corrects = [None] + [BatchedTensorStorage(len(loader), exp_data.batch_size) for loader in
							   dataset.data_loaders[1:]]

	epoch_activations = [BatchedTensorStorage(len(loader), exp_data.batch_size, (penultimate_layer.in_features,),
											  lazy=(loader.dataset.name == 'training' and exp_data.loss == 'Contrastive')) for loader in
						 dataset.data_loaders]
	dataloader_index = [0]

	def store_activations(m, i, o):
		if dataloader_index[0] > 0 or exp_data.loss == 'Contrastive':
			epoch_activations[dataloader_index[0]].append(squeeze(i[0]))

	penultimate_layer.register_forward_hook(store_activations)

	return epoch_corrects, epoch_activations, dataloader_index


def train(model_objects, dataset, criterion, exp_data):

	epoch_corrects, epoch_activations, dataloader_index = setup_activation_collection(model_objects, dataset, exp_data)

	num_training_batches = 1000
	total_num_batches = num_training_batches + sum([len(l) for l in dataset.data_loaders[1:]])

	for epoch in tqdm(range(exp_data.epochs_completed, exp_data.max_epochs),
					  desc='Training epochs completed',
					  file=sys.stdout):
		with tqdm(total=total_num_batches, leave=False, file=sys.stdout) as pbar:

			dataloader_index[0] = 0
			[batched_storage_tensor.reset() for batched_storage_tensor in epoch_corrects[1:]]
			[batched_storage_tensor.reset() for batched_storage_tensor in epoch_activations]

			training_loss, training_accuracy = run_epoch(model_objects,
														 dataset.train_loader,
														 exp_data.loss,
														 criterion,
														 True,
														 pbar,
														 num_batches=num_training_batches,
														 batch_activations=epoch_activations[0] if exp_data.loss == "Contrastive" else None)

			exp_data.eval_data.training_losses.append(training_loss)
			exp_data.eval_data.training_accuracies.append(np.mean(training_accuracy))

			run_evalutation(model_objects, dataset, criterion, exp_data, pbar, epoch_corrects, epoch_activations, dataloader_index)

		exp_data.eval_data.validation_and_partial_base_accuracies.append(((exp_data.eval_data.full_validation_accuracies[-1] * len(exp_data.full_instances)) + exp_data.eval_data.partial_base_accuracies[-1] * len(exp_data.partial_instances)) / (len(exp_data.full_instances) + len(exp_data.partial_instances)))
		# exp_data.eval_data.validation_and_partial_base_accuracies.append(None)
		exp_data.eval_data.save()

		max_accuracy_index = np.argmax(exp_data.eval_data.validation_and_partial_base_accuracies)

		if max_accuracy_index == len(exp_data.eval_data.validation_and_partial_base_accuracies) - 1:
			for corrects, activations, epoch_correct, epoch_activation in zip(exp_data.eval_data.corrects,
																			  exp_data.eval_data.activations,
																			  epoch_corrects[1:],
																			  epoch_activations[1:]):
				corrects.arr = epoch_correct.all_data().cpu().numpy()
				activations.arr = epoch_activation.all_data().cpu().numpy()

			model_objects.save(True)

		model_objects.save(False)

		exp_data.epochs_completed = epoch + 1
		exp_data.save()

		num_epochs_after_peak = 7
		if (exp_data.epochs_completed >= (exp_data.min_epochs + num_epochs_after_peak)) and (max_accuracy_index < (len(exp_data.eval_data.validation_and_partial_base_accuracies) - num_epochs_after_peak)):
			break

	for corrects, activations in zip(exp_data.eval_data.corrects, exp_data.eval_data.activations):
		corrects.dump()
		activations.dump()


def evaluate(model_objects, dataset, criterion, exp_data, collect_for_deephys=False):
	epoch_corrects, epoch_activations, dataloader_index = setup_activation_collection(model_objects, dataset, exp_data)
	with tqdm(total=sum([len(l) for l in dataset.data_loaders[1:]]), leave=False, file=sys.stdout) as pbar:
		run_evalutation(model_objects, dataset, criterion, exp_data, pbar, epoch_corrects, epoch_activations, dataloader_index, collect_for_deephys)

	if not collect_for_deephys:
		for corrects, activations, epoch_correct, epoch_activation in zip(exp_data.eval_data.corrects,
																		  exp_data.eval_data.activations,
																		  epoch_corrects[1:],
																		  epoch_activations[1:]):

			corrects.arr = epoch_correct.all_data().cpu().numpy()
			activations.arr = epoch_activation.all_data().cpu().numpy()

			corrects.dump()
			activations.dump()

