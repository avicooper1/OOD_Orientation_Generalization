import os
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
import json
from ast import literal_eval

import importlib.util
torch_avail = importlib.util.find_spec("torch")
if importlib.util.find_spec("torch"):
    from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('/home/avic/OOD_Orientation_Generalization')
from dataset_path import DatasetPathOLD, DatasetType


def format_vars(obj):
    out = ''
    obj_vars = vars(obj)
    for k in obj_vars:
        out += f'{k} : {obj_vars[k]}\n'
    return out

@dataclass
class PersistentDataClass:

    def save(self):
        with open(self.store_path, 'w') as f:
            json.dump(vars(self), f, ensure_ascii=False)

    def load(self):
        if os.path.exists(self.store_path):
            with open(self.store_path, 'r') as f:
                data = json.load(f)
            for key, val in data.items():
                setattr(self, key, val)




@dataclass
class DatasetPath(PersistentDataClass):
    root_path: str
    model_category: str
    resolution: int
    complete: bool
    images_compressed: bool

    def __post_init__(self):
        self.path = self.__repr__()
        self.image_dir = os.path.join(self.path, 'images')
        self.store_path = os.path.join(self.path, 'configuration.json')

        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

    def __repr__(self):
        return os.path.join(self.root_path, f'datasets{self.resolution}x{self.resolution}', self.model_category)

    def annotation_path(self, i):
        return os.path.join(self.__repr__(), f'{self.model_category}{i}_dataset.csv')

    def annotation_file(self, i):
        df_file_path = self.annotation_path(i)
        if os.path.exists(df_file_path):
            df = pd.read_csv(df_file_path, converters={"cubelet_i": literal_eval}, delimiter=';')
            df.cubelet_i = df.cubelet_i.apply(np.array)
            return df

    # This function checks if a specific subdataset is finished rendering. It will return the index of the next cubelet to be rendered, or -1 if all cubelets are finished
    def check_annotation_file(self, i):
        df = self.annotation_file(i)
        if df is not None and len(df) > 0:
            last_cubelet = df.cubelet_i.iloc[-1]
            if np.all(last_cubelet == self.resolution - 1):
                return -1
            else:
                return np.ravel_multi_index(last_cubelet, (self.resolution, self.resolution, self.resolution))
        return 0

    def annotations_paths(self):
        return [self.annotation_path(i) for i in range(50)]

    def merged_annotation_path(self):
        return os.path.join(self.__repr__(), 'merged_dataset.csv')

    def image_file_name(self, model_i, cubelet_i):
        return os.path.join(self.image_dir, f'{self.model_category}_{model_i}_{cubelet_i}.png')

    @classmethod
    def get_dataset(cls, root_path, i, resolution):
        return cls(root_path, ['plane', 'car', 'lamp', 'SM'][i], resolution, False, False)

# @dataclass
# class AnnotationFrame:
#     dataset_path: DatasetPath
#     i: int
#
#     def __post_init__(self):
#         self.load()
#
#     def load(self):
#         df_file_path = self.dataset_path.annotation_path(self.i)
#         if os.path.exists(df_file_path):
#             df = pd.read_csv(df_file_path, converters={"cubelet_i": literal_eval})
#             df.cubelet_i = df.cubelet_i.apply(np.array)
#             df['image_name_without_directory'] = df.image_name
#             df.image_name = df.image_name.apply(lambda n: os.path.join(self.image_dir, n))
#             self.df = df
#         else:
#             self.df = None
#
#     def save(self):




class ModelType(Enum):
    ResNet = 0 #Batch size: 230, Learning rate: 0.001
    DenseNet = 1 #Batch size is 75,
    Inception = 2
    CorNet = 3
    ViT = 4
    Equivariant = 5
    DeiT = 6


@dataclass
class ExpData:
    job_id: int
    data_div: int
    model_type: ModelType
    pretrained: bool
    num: str
    training_category: str
    testing_category: str
    hole: int
    augment: bool
    scale: bool
    restriction_axes: tuple
    lr: float
    batch_size: int
    max_epoch: int

    def __post_init__(self):
        self.name = f'Div{self.data_div}'
        self.dir = f'/home/avic/om2/experiments/exp{self.num}'
        self.logs_dir = os.path.join(self.dir, 'logs')
        self.eval_dir = os.path.join(self.dir, 'eval')
        self.stats_dir = os.path.join(self.dir, 'stats')
        self.checkpoints_dir = os.path.join(self.dir, 'checkpoints')
        self.tensorboard_logs_dir = os.path.join(self.dir, 'tensorboard_logs', str(self.job_id))
        for p in (self.logs_dir, self.eval_dir, self.stats_dir, self.checkpoints_dir, self.tensorboard_logs_dir):
            os.makedirs(p, exist_ok=True)
        self.logs = os.path.join(self.logs_dir, f'{self.name}.txt')
        self.eval = os.path.join(self.eval_dir, f'{self.name}.csv')
        self.testing_frame_path = os.path.join(self.eval_dir, f'TestingFrame_{self.name}.csv')
        self.stats = os.path.join(self.stats_dir, f'{self.name}.csv')
        self.checkpoint = os.path.join(self.checkpoints_dir, f'{self.name}.pt')
        self.eval_heatmap = os.path.join(self.eval_dir, f'{self.name}_heatmap.npy')
        self.eval_heatmap_id = os.path.join(self.eval_dir, f'{self.name}_heatmap_id.npy')
        self.eval_heatmap_ood = os.path.join(self.eval_dir, f'{self.name}_heatmap_ood.npy')
        self.image_activations = os.path.join(self.eval_dir, f'{self.name}_image_activations.npy')
        self.activations_heatmap = os.path.join(self.eval_dir, f'{self.name}_activations_heatmap.npy')

    def __str__(self):
        return self.__repr__()

    def __repr__(self, print=False):
        return format_vars(self) if print else ''

    def log(self, lines):
        with open(self.logs, 'a') as f:
            if type(lines) is str:
                print(lines)
                f.write(lines)
            else:
                for line in lines:
                    print(line)
                    f.write(f'{line}\n\n')

    # For each of the 5 models:
    # For each of the 4 synthetic datasets
    # For each of not-pretrained and pretrained
    # For each of scaled and not-scaled
    # For each of augmented and non-augmented
    # For each of (3 axes + XY hole + XY bin and hole) (5 total)
    # For each data div (4 total)
    # 640 total right now
    @classmethod
    def get_experiments(cls, i):
        category = ['plane', 'SM', 'car', 'lamp'][(i % 640) // 160]
        axes_i = (i % 40) // 8
        model_i = i // 640
        ret = cls(
            job_id=i,
            data_div=(10 * (i % 4)) + 10,
            model_type=ModelType(model_i),
            pretrained=((i // 80) % 2) == 1,
            num=i // 4,
            training_category=category,
            testing_category=category,
            augment=((i // 4) % 2) == 1,
            hole=max(0, axes_i - 2),
            scale=((i // 40) % 2) == 1,
            restriction_axes=((0, 1), (1, 2), (0, 2))[axes_i if axes_i <= 2 else 0],
            lr=[0.001, 0.001, 0.001, 0.0001, 0.0001, None, 0.001][model_i],
            batch_size=[230, 64, 90, 128, 52, 128, 52][model_i],
            max_epoch=[13,10,10,10,25,10,10][model_i])
        return None if ret.pretrained and not (ret.model_type == ModelType.ResNet or ret.model_type == ModelType.CorNet) else ret


@dataclass
class _Ann:
    exp_data: ExpData
    category: str
    dataset_type: DatasetType

    def __post_init__(self):
        self.path = DatasetPathOLD(self.category, self.dataset_type, self.exp_data.scale, self.exp_data.restriction_axes)
        self.ann = pd.read_csv(self.path.merged_annotation_path())
        self.models = self.ann.model_name.unique()

    def __repr__(self):
        return format_vars(self)


@dataclass
class Ann:
    exp_data: ExpData
    training: bool
    hole: bool = False

    def __post_init__(self):
        self.category = self.exp_data.training_category if self.training else self.exp_data.testing_category

        self.full = _Ann(self.exp_data, self.category, DatasetType.Full)
        self.bin = _Ann(self.exp_data, self.category, DatasetType.Bin if not self.hole else DatasetType.Hole)

    def __repr__(self):
        return format_vars(self)



# @dataclass
# class Splits:
#     training: DataLoader
#     testing: DataLoader
#     training_validation: DataLoader
#     held_validation: DataLoader
#     names = ('Training', 'Testing', 'Training Validation', 'Held Validation')
#
#     def __post_init__(self):
#         self.all_loaders = (self.training, self.testing, self.training_validation, self.held_validation)
#         if self.held_validation is not None:
#             self.all_loaders_named = list(zip(self.all_loaders, self.names))
#         else:
#             self.all_loaders_named = list(zip(self.all_loaders[:-1], self.names[:-1]))
#
#     def __repr__(self):
#         out = ''
#         for loader, name in self.all_loaders_named:
#             out += f'{len(loader.dataset)} {name} images \n'
#         return out
