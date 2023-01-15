import sys
from torch import cuda, vstack
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, WeightedRandomSampler, SequentialSampler
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from utils.persistent_data_class import *
from utils.tools import get_base_mask
from tqdm.contrib.concurrent import process_map
from torch import tensor, empty
import torch.multiprocessing as mp


def get_dataloader(dataset, shuffle, batch_size):

	if shuffle:
		dataset_sampler = WeightedRandomSampler(dataset.frame.weights, len(dataset.frame), True)
	else:
		dataset_sampler = SequentialSampler(dataset)

	return DataLoader(dataset=dataset,
					  sampler=BatchSampler(dataset_sampler, batch_size, drop_last=False))


def perform_transforms(args):
	images, image_idx, random_affine_transform = args
	images[image_idx] = random_affine_transform(images[image_idx])


def get_image_group(group_path):
	return group_path, Arr.from_path(group_path).load()


@dataclass
class RotationDataset:

	exp_data: ExpData
	num_workers: int = 8
	preload_dataset: bool = True
	pool: mp.Pool = None

	def __post_init__(self):

		for frame in self.exp_data.frames:
			if frame.df is None:
				self.setup_frames()
				break

		self.instance_codec = {instance: i for i, instance in enumerate(self.exp_data.full_instances + self.exp_data.held_instances + self.exp_data.partial_instances)}

		group_cache = {}

		if self.preload_dataset:

			mp.set_start_method('spawn')
			self.pool = mp.Pool(self.num_workers)

			for group_path, arr in process_map(get_image_group,
											   self.exp_data.training_frame.df.image_group.unique(),
											   # np.concatenate((self.exp_data.training_frame.df.image_group.unique(),
												# 			   self.exp_data.partial_ood_frame.df.image_group.unique())),
											   max_workers=self.num_workers,
											   file=sys.stdout,
											   desc=f'Loading image groups from storage'):
				group_cache[group_path] = arr

		self.datasets = [_Dataset(name, self, frame.df, self.preload_dataset, group_cache, self.pool)
						 for frame, name in zip(self.exp_data.frames, (['training',
																		'full validation',
																		'partial base',
																		'partial OOD']))]

		del group_cache

		self.data_loaders = [get_dataloader(dataset, shuffle, self.exp_data.batch_size)
							 for dataset, shuffle in zip(self.datasets, (True, False, False, False, False, False))]

		self.train_loader, self.full_validation_loader, self.partial_base_loader, self.partial_ood_loader = self.data_loaders

		self.eval_loaders = self.data_loaders[1:]

	def setup_frames(self):

		assert self.exp_data.epochs_completed == 0, 'Attempting to regenerate training and evaluation frames after training'

		if self.exp_data.num_fully_seen == 50:
			dataset_path = ImageDataset(self.exp_data.path,
										self.exp_data.full_category,
										self.exp_data.dataset_resolution)

			instances_frame = dataset_path.merged_annotation_file.df

			self.exp_data.full_instances = instances_frame.instance_name.unique().tolist()
			self.exp_data.held_instances = []
			self.exp_data.partial_instances = []

			self.exp_data.save()

			grouper = instances_frame.groupby([(instances_frame.cubelet_i // 2).astype(str), instances_frame.instance_name],
											  sort=False)

			self.exp_data.full_validation_frame.df = grouper.sample(n=1)
			self.exp_data.training_frame.df = instances_frame[~instances_frame.index.isin(self.exp_data.full_validation_frame.df.index)].copy()
			self.exp_data.training_frame.df['weights'] = 1 / len(self.exp_data.training_frame)
			self.exp_data.partial_base_frame.df = self.exp_data.full_validation_frame.df.sample(n=0)
			self.exp_data.partial_ood_frame.df = self.exp_data.full_validation_frame.df.sample(n=0)

		else:
			full_dataset_path = ImageDataset(self.exp_data.path,
											 self.exp_data.full_category,
											 self.exp_data.dataset_resolution)
			partial_dataset_path = ImageDataset(self.exp_data.path,
												self.exp_data.partial_category,
												self.exp_data.dataset_resolution)

			full_instances_frame = full_dataset_path.merged_annotation_file.df
			partial_instances_frame = partial_dataset_path.merged_annotation_file.df

			instance_names = np.random.permutation(full_instances_frame.instance_name.unique()).tolist()
			self.exp_data.full_instances = instance_names[:self.exp_data.num_fully_seen]
			self.exp_data.held_instances = instance_names[self.exp_data.num_fully_seen:40]

			if self.exp_data.full_category == self.exp_data.partial_category:
				self.exp_data.partial_instances = instance_names[40:]
			else:
				self.exp_data.partial_instances = np.random.permutation(partial_instances_frame.instance_name.unique()).tolist()[:10]

			self.exp_data.save()

			full_instances_frame = full_instances_frame[full_instances_frame.instance_name.isin(self.exp_data.full_instances)].copy()
			partial_instances_frame = partial_instances_frame[partial_instances_frame.instance_name.isin(self.exp_data.partial_instances)].copy()

			grouper = full_instances_frame.groupby([(full_instances_frame.cubelet_i // 2).astype(str), full_instances_frame.instance_name], sort=False)

			self.exp_data.full_validation_frame.df = grouper.sample(n=1)

			full_training_frame = full_instances_frame[~full_instances_frame.index.isin(self.exp_data.full_validation_frame.df.index)].copy()
			full_training_frame['weights'] = 1 / len(full_training_frame)

			partial_base_mask = get_base_mask(partial_instances_frame, self.exp_data.base_orientations)

			self.exp_data.partial_base_frame.df = partial_instances_frame[partial_base_mask].copy()
			self.exp_data.partial_base_frame.df['weights'] = 1 / len(self.exp_data.partial_base_frame.df)
			self.exp_data.partial_ood_frame.df = partial_instances_frame[~partial_base_mask].copy()

			if len(self.exp_data.held_instances) > 0:

				held_instances_frame = full_dataset_path.merged_annotation_file.df
				held_instances_frame = held_instances_frame[held_instances_frame.instance_name.isin(self.exp_data.held_instances)].copy()
				held_base_mask = get_base_mask(held_instances_frame, self.exp_data.base_orientations)
				held_frame = held_instances_frame[held_base_mask].copy()
				held_frame['weights'] = 1 / len(held_frame)

				# self.exp_data.training_frame.df = pd.concat([full_training_frame])
				self.exp_data.training_frame.df = pd.concat(
					[full_training_frame, held_frame, self.exp_data.partial_base_frame.df])

			else:
				self.exp_data.training_frame.df = pd.concat([full_training_frame, self.exp_data.partial_base_frame.df])
				# self.exp_data.training_frame.df = pd.concat([full_training_frame])

		for frame in self.exp_data.frames:
			frame.dump()


@dataclass
class _Dataset(Dataset):
	name: str
	dataset: RotationDataset
	frame: pd.DataFrame
	preload_dataset: bool
	group_cache: {str: object} = None
	pool: mp.Pool = None
	loaded_dataset: cuda.ByteTensor = None
	loaded_targets: cuda.LongTensor = None
	current_batch: tensor = None

	def __post_init__(self):
		self.frame.reset_index(drop=True, inplace=True)

		# Note: During dataset generation / rendering, images are guaranteed to have 20 pixels of padding.
		# Images are 224x224 pixels. 20 / 224 ~ 0.089, which allows for up to this much translation without losing
		# parts of the object
		# TODO We do implement scaling. Are we ok if parts of the image might be lost? Confirm these parameters
		self.pad_transform = transforms.Pad(15)
		self.random_affine_transform = transforms.RandomAffine(degrees=(-180, 180) if self.dataset.exp_data.augment and (self.name == 'training') else 0,
															translate=(0.08, 0.08),
															scale=(0.8, 1.2),
															interpolation=InterpolationMode.BILINEAR)

		if self.preload_dataset:

			image_shape = next(iter(self.group_cache.values())).shape[-1]

			self.loaded_dataset = cuda.ByteTensor(len(self.frame), image_shape, image_shape)
			self.loaded_targets = cuda.LongTensor(len(self.frame)).cuda()

			for (group_path, instance_name), group in tqdm(self.frame.groupby(['image_group', 'instance_name']),
										  file=sys.stdout,
										  desc=f'Sending {self.name} dataset to GPU'):
				self.loaded_dataset[group.index] = cuda.ByteTensor(self.group_cache[group_path][group.image_idx.values])
				self.loaded_targets[group.index] = self.dataset.instance_codec[instance_name]

			del self.group_cache

		self.batch = empty((self.dataset.exp_data.batch_size, 1, 224, 224)).cuda().share_memory_()

	def __len__(self):
		return len(self.frame)

	def __getitem__(self, idx):

		if self.preload_dataset:
			images = self.loaded_dataset[idx].float() / 255
		else:
			images = (vstack([read_image(name) for name in self.frame.iloc[idx].image_name]) / 255).cuda()

		images = images.unsqueeze(axis=1)
		self.batch[:len(idx)] = transforms.Pad(15)(images)

		self.pool.imap_unordered(perform_transforms,
								 ((self.batch, image_idx, self.random_affine_transform) for image_idx in
								  range(len(idx))), chunksize=4)

		self.batch[:] = (self.batch - self.batch.mean((-2, -1)).unsqueeze(2).unsqueeze(3)) / self.batch.std(
			(-2, -1)).unsqueeze(2).unsqueeze(3)

		return self.batch[:len(idx)], self.loaded_targets[idx]
