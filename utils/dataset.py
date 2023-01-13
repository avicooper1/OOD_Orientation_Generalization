import sys
from torch import unsqueeze
from torch import cuda, vstack, Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, WeightedRandomSampler, SequentialSampler
from torchvision import transforms
from torchvision.io import read_image
from sklearn.preprocessing import LabelEncoder
from torchvision.transforms import InterpolationMode
from torch.nn import Sequential
from tqdm import tqdm
from utils.persistent_data_class import *
from utils.tools import get_base_mask
from tqdm.contrib.concurrent import process_map
from torch import tensor


def get_dataloader(dataset, shuffle, batch_size):

	if shuffle:
		dataset_sampler = WeightedRandomSampler(dataset.frame.weights, len(dataset.frame), True)
	else:
		dataset_sampler = SequentialSampler(dataset)

	# return DataLoader(dataset=dataset,
	# 				  sampler=BatchSampler(dataset_sampler, batch_size, drop_last=False))

	return DataLoader(dataset=dataset,
					  sampler=dataset_sampler,
					  batch_size=batch_size,
					  drop_last=False,
					  num_workers=4)


def get_image_group(group_path):
	return group_path, Arr.from_path(group_path).load()


@dataclass
class RotationDataset:

	exp_data: ExpData
	preload_dataset: bool = True

	def __post_init__(self):

		for frame in self.exp_data.frames:
			if frame.df is None:
				self.setup_frames()
				break

		self.instance_codec = {instance: i for i, instance in enumerate(self.exp_data.full_instances + self.exp_data.held_instances + self.exp_data.partial_instances)}

		group_cache = {}

		if self.preload_dataset:

			for group_path, arr in process_map(get_image_group,
											   self.exp_data.training_frame.df.image_group.unique(),
											   # np.concatenate((self.exp_data.training_frame.df.image_group.unique(),
												# 			   self.exp_data.partial_ood_frame.df.image_group.unique())),
											   max_workers=4,
											   file=sys.stdout,
											   desc=f'Loading image groups from storage'):
				group_cache[group_path] = arr

		self.datasets = [_Dataset(name, self, frame.df, self.preload_dataset, group_cache)
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
	loaded_dataset: cuda.ByteTensor = None

	def __post_init__(self):
		self.frame.reset_index(drop=True, inplace=True)

		# Note: During dataset generation / rendering, images are guaranteed to have 20 pixels of padding.
		# Images are 224x224 pixels. 20 / 224 ~ 0.089, which allows for up to this much translation without losing
		# parts of the object
		# TODO We do implement scaling. Are we ok if parts of the image might be lost? Confirm these parameters
		self.transform = Sequential(transforms.Pad(15),
									transforms.RandomAffine(degrees=(-180, 180) if self.dataset.exp_data.augment and (self.name == 'training') else 0,
															translate=(0.08, 0.08),
															scale=(0.8, 1.2),
															interpolation=InterpolationMode.BILINEAR))

		if self.preload_dataset:

			image_shape = next(iter(self.group_cache.values())).shape[-1]

			self.loaded_dataset = cuda.ByteTensor(len(self.frame), image_shape, image_shape).share_memory_()

			for group_path, group in tqdm(self.frame.groupby('image_group'),
										  file=sys.stdout,
										  desc=f'Sending {self.name} dataset to GPU'):
				self.loaded_dataset[group.index] = cuda.ByteTensor(self.group_cache[group_path][group.image_idx.values])

			del self.group_cache

	def get_img(self, idx):
		if self.preload_dataset:
			images = self.loaded_dataset[idx].float() / 255
		else:
			images = (vstack([read_image(name) for name in self.frame.iloc[idx].image_name]) / 255).cuda()



		images = self.transform(images.unsqueeze(0))
		images = transforms.Normalize(images.mean(), images.std())(images)
		return images

		# images = vstack([self.transform(image.unsqueeze(axis=0)) for image in images])
		# images = transforms.Normalize(tuple(images.mean((-2, -1))), tuple(images.std((-2, -1))))(images)
		# return images.unsqueeze(axis=1)

	def __len__(self):
		return len(self.frame)

	def __getitem__(self, idx):
		return self.get_img(idx), self.dataset.instance_codec[self.frame.iloc[idx].instance_name]
