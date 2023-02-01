import os.path
from dataclasses import dataclass, field, is_dataclass
import pandas as pd
from ast import literal_eval
import json
from utils.persistent_data_object import *


def format_vars(obj):
	out = ''
	obj_vars = vars(obj)
	for k in obj_vars:
		out += f'{k} : {obj_vars[k]}\n'
	return out


PDC_KEY = '__PersistentDataClass__'
PDO_KEY = '__PersistentDataObject__'


class ProjectJSONEncoder(json.JSONEncoder):
	def default(self, o):
		if isinstance(o, PersistentDataObject):
			return {PDO_KEY: [type(o).__name__, o.file_path]}
		if isinstance(o, PersistentDataClass):
			return
		# return {PDC_KEY: [type(o).__name__, o.file_path]}
		return super().default(o)


def instantiate_data_object_from_str(s):
	match s:
		case 'PersistentDataObject':
			return PersistentDataObject
		case 'Arr':
			return Arr
		case 'DF':
			return DF
		case 'AnnotationFile':
			return AnnotationFile
	print("Should not reach here")
	exit(-1)


def ProjectJSONDecoder(dct):
	if PDO_KEY in dct:
		values = dct[PDO_KEY]
		return instantiate_data_object_from_str(values[0]).from_path(values[1])
	return dct


@dataclass
class PersistentDataClass:

	def __post_init__(self):
		if self.on_disk():
			self.load()
		else:
			if hasattr(self, 'initial_init'):
				self.initial_init()
			self.save()
		if hasattr(self, 'finish_init'):
			self.finish_init()

	def on_disk(self):
		return os.path.exists(self.store_path)

	def save(self):
		with open(self.store_path, 'w') as f:
			json.dump(vars(self), f, ensure_ascii=False, cls=ProjectJSONEncoder, indent=4)

	def load(self):
		if os.path.exists(self.store_path):
			with open(self.store_path, 'r') as f:
				data = json.load(f, object_hook=ProjectJSONDecoder)
				for key, val in data.items():
					setattr(self, key, val)


@dataclass
class ImageDataset(PersistentDataClass):
	storage_path: str
	model_category: str
	resolution: int
	rendering_complete: bool = False
	images_compressed: bool = False
	images_grouped: bool = False

	def __post_init__(self):
		self.path = os.path.join(self.storage_path, f'datasets{self.resolution}x{self.resolution}', self.model_category)
		self.store_path = os.path.join(self.path, 'configuration.json')
		super().__post_init__()

	def initial_init(self):
		os.makedirs(self.path, exist_ok=True)

	def finish_init(self):
		self.merged_annotation_file = AnnotationFile(self.path, 'merged_dataset', storage_path=self.storage_path)
		self.subdatasets = [ImageSubDataset(self.storage_path, self.path, self.model_category, i, self.resolution) for i in range(50)]

	def __repr__(self):
		return self.path

	def __check_attribute(self, attribute, on_callback, message):
		remaining = list(filter(attribute, self.subdatasets))
		if len(remaining) == 0:
			on_callback()
			return True
		print(f'Several subdatasets did not pass: {message}. They are:')
		for r in remaining:
			print(r.model_category, r.model_i)
		return False

	def finalize_merging(self):

		def concat_subdataset_dfs():
			merged_dataset = pd.concat([sd.annotation_file.df for sd in self.subdatasets]).reset_index(drop=True)
			merged_dataset.image_name = merged_dataset.image_name.apply(lambda img: os.path.join(*img.split(os.sep)[-5:]))
			merged_dataset['image_group'] = merged_dataset.image_name.apply(lambda p: os.path.join(*p.split(os.sep)[-5:-2], 'group.npy.gz'))
			merged_dataset.cubelet_i = merged_dataset.cubelet_i.apply(lambda ci: tuple(ci))
			merged_dataset['image_idx'] = merged_dataset.image_name.apply(lambda image_name: image_name.split(os.sep)[-1][5:-4])
			merged_dataset.to_csv(self.merged_annotation_file.file_path, sep=';', index=False)
			self.rendering_complete = True
			self.save()

		def save_images_compressed():
			self.images_compressed = True
			self.save()

		def save_images_grouped():
			self.images_grouped = True
			self.save()

		if not self.rendering_complete:
			if not self.__check_attribute(lambda sd: not sd.rendering_complete, concat_subdataset_dfs, 'rendering complete'):
				return False
		if not self.images_compressed:
			if not self.__check_attribute(lambda sd: not sd.images_compressed, save_images_compressed, 'images compressed'):
				return False
		if not self.images_grouped:
			if not self.__check_attribute(lambda sd: not sd.images_grouped, save_images_grouped, 'images grouped'):
				return False
		return True


@dataclass
class ImageSubDataset(PersistentDataClass):
	storage_path: str
	dataset_path: str
	model_category: str
	model_i: int
	resolution: int
	rendering_complete: bool = False
	images_compressed: bool = False
	images_grouped: bool = False

	def __post_init__(self):
		self.path = os.path.join(self.dataset_path, f'{self.model_category}{self.model_i}_dataset')
		self.store_path = os.path.join(self.path, 'configuration.json')
		super().__post_init__()

	def initial_init(self):
		os.makedirs(self.path, exist_ok=True)
		self.image_dir = os.path.join(self.path, 'images')
		self.image_group = Arr(self.path, 'group', compress=True)

	def finish_init(self):
		self.annotation_file = AnnotationFile(self.path, 'dataset', storage_path=self.storage_path)

	def __repr__(self):
		return self.path

	def image_path(self, cubelet_i):
		return os.path.join(os.path.join(*self.path.split(os.sep)[-3:]), 'images', f'image{cubelet_i}.png')


@dataclass
class ExpData(PersistentDataClass):
	path: str
	num: int
	run: int
	num_fully_seen: int
	model_type: str
	hook_layer: str
	full_category: str
	partial_category: str
	dataset_resolution: int
	loss: str
	pretrained: bool
	augment: bool
	sliver: bool
	half_data: bool
	free_axis: str = None
	base_orientations: [[[float]]] = None
	lr: float = 0.01
	batch_size: int = 128
	min_epochs: int = 100
	max_epochs: int = 250
	full_instances: [str] = None
	held_instances: [str] = None
	partial_instances: [str] = None
	epochs_completed: int = 0
	create_exp: bool = False
	complete: bool = False

	def __post_init__(self):

		self.dir = os.path.join(self.path, f'experiments/exp{self.num}/{str(self.num_fully_seen) + "_fully_seen" if self.partial_category != "all_fully_seen" else self.full_category}/run{self.run}')
		self.store_path = os.path.join(self.dir, 'configuration.json')

		assert os.path.exists(self.store_path) or self.create_exp, 'Parameters disallow creating this experiment'

		super().__post_init__()

	def initial_init(self):
		self.log = os.path.join(self.dir, 'log.txt')
		self.checkpoint = os.path.join(self.dir, 'checkpoint.pt')
		self.peak_checkpoint = os.path.join(self.dir, 'peak_checkpoint.pt')

		self.frames_dir = os.path.join(self.dir, 'frames')
		os.makedirs(self.frames_dir, exist_ok=True)

		self.eval_dir = os.path.join(self.dir, 'eval')
		os.makedirs(self.eval_dir, exist_ok=True)

	def finish_init(self):
		self.training_frame = AnnotationFile(self.frames_dir, 'training', storage_path=self.path)
		self.full_validation_frame = AnnotationFile(self.frames_dir, 'full_validation', storage_path=self.path)
		self.partial_base_frame = AnnotationFile(self.frames_dir, 'partial_base', storage_path=self.path)
		self.partial_ood_frame = AnnotationFile(self.frames_dir, 'partial_ood', storage_path=self.path)

		self.frames = [self.training_frame,
					   self.full_validation_frame,
					   self.partial_base_frame,
					   self.partial_ood_frame,]

		self.eval_data = EvalData(self.eval_dir)


	@classmethod
	def from_job_i(cls, project_path, storage_path, job_i, num_runs=5, create_exp=False):
		return cls.from_num(project_path, storage_path, job_i // 20, ((job_i % 4) + 1) * 10,
							(job_i % (num_runs * 4)) // 4, create_exp)

	@classmethod
	def from_num(cls, project_path, storage_path, exp_num, num_fully_seen, run, create_exp=False):

		exps = pd.read_csv(os.path.join(project_path, 'exps.csv'), comment='#')
		exp = exps.iloc[exp_num]

		if exp.full_category == 'all_fully_seen':

			base_orientations = None
			full_category = ['plane', 'SM', 'car', 'lamp'][(num_fully_seen // 10) - 1]
			num_fully_seen = 50

		else:

			full_category = exp.full_category

			assert (type(exp.base_orientations) is str) ^ (type(exp.free_axis) is str), 'For each experiment, either a set of orientations or a free axis can be specified, but not both'

			if type(exp.base_orientations) is str:
				base_orientations = literal_eval(exp.base_orientations)
			else:
				match exp.free_axis:
					case 'γ':
						base_orientations = [[[-5, 5]], [[-0.1, 0.1]], [[-0.25, 0.25]]]
					case 'β':
						base_orientations = [[[-0.25, 0.25]], [[-5, 5]], [[-0.25, 0.25]]]
					case 'α':
						base_orientations = [[[-0.25, 0.25]], [[-0.1, 0.1]], [[-5, 5]]]
					case 'hole':
						base_orientations = [[[-1.8, -1.3]], [[-0.1, 0.1]], [[-5, 5]]]

		def parse_flag(f):
			return (not pd.isna(f)) and bool(f)

		return cls(storage_path,
				   exp_num,
				   run,
				   num_fully_seen,
				   exp.model_type,
				   exp.hook_layer,
				   full_category,
				   exp.partial_category,
				   int(exp.dataset_resolution),
				   exp.loss,
				   parse_flag(exp.pretrained),
				   parse_flag(exp.augment),
				   parse_flag(exp.sliver),
				   parse_flag(exp.half_data),
				   exp.free_axis if type(exp.free_axis) == str else None,
				   base_orientations,
				   create_exp=create_exp)

	def log(self, lines):
		with open(self.logs, 'a') as f:
			if type(lines) is str:
				print(lines)
				f.write(lines)
			else:
				for line in lines:
					print(line)
					f.write(f'{line}\n\n')


@dataclass
class EvalData(PersistentDataClass):
	path: str

	training_accuracies: [float] = field(default_factory=list)
	full_validation_accuracies: [float] = field(default_factory=list)
	partial_base_accuracies: [float] = field(default_factory=list)
	partial_ood_accuracies: [float] = field(default_factory=list)

	training_losses: [float] = field(default_factory=list)
	full_validation_losses: [float] = field(default_factory=list)
	partial_base_losses: [float] = field(default_factory=list)
	partial_ood_losses: [float] = field(default_factory=list)

	validation_and_partial_base_accuracies: [float] = field(default_factory=list)

	def __post_init__(self):
		self.store_path = os.path.join(self.path, 'configuration.json')
		super().__post_init__()

	def initial_init(self):

		self.full_validation_correct = Arr(self.path, 'full_validation_correct', compress=True)
		self.partial_base_correct = Arr(self.path, 'partial_base_correct', compress=True)
		self.partial_ood_correct = Arr(self.path, 'partial_ood_correct', compress=True)

		self.full_validation_activations = Arr(self.path, 'full_validation_activations', compress=True)
		self.partial_base_activations = Arr(self.path, 'partial_base_activations', compress=True)
		self.partial_ood_activations = Arr(self.path, 'partial_ood_activations', compress=True)


	def finish_init(self):
		self.accuracies = [self.full_validation_accuracies,
						   self.partial_base_accuracies,
						   self.partial_ood_accuracies]

		self.losses = [self.full_validation_losses,
					   self.partial_base_losses,
					   self.partial_ood_losses]

		self.corrects = [self.full_validation_correct,
						 self.partial_base_correct,
						 self.partial_ood_correct]

		self.activations = [self.full_validation_activations,
							self.partial_base_activations,
							self.partial_ood_activations]
