import os
from dataclasses import dataclass
from enum import Enum

class DatasetType(Enum):
    Full = 0
    Bin = 1
    Hole = 2
    NonBin = 3

@dataclass
class DatasetPathOLD:
    model_category: str
    type: DatasetType
    scale: bool
    restriction_axes: tuple

    def __post_init__(self):
        _restriction_axes_names = ('X', 'Y', 'Z')
        self.restriction_axes_named = tuple((_restriction_axes_names[ra] for ra in self.restriction_axes))
        self.path = self.__repr__()
        self.image_path = os.path.join(self.path, 'images')
        self.config_file = os.path.join(self.path, 'configuration.txt')

    def __repr__(self):
        p = os.path.join('/home/avic/om2/datasets', self.model_category)

        if self.type == DatasetType.Full:
            p = os.path.join(p, 'full')
        elif self.type == DatasetType.Bin or self.type == DatasetType.Hole:
            p = os.path.join(p, 'bin')
        elif self.type == DatasetType.NonBin:
            p = os.path.join(p, 'non_bin')

        if self.scale:
            p = os.path.join(p, 'scaled' if self.type == DatasetType.Full else 'mid_scaled')
        if not self.type == DatasetType.Full:
            p = os.path.join(p, f'{self.restriction_axes_named[0]}_{self.restriction_axes_named[1]}')
        if self.type == DatasetType.Hole:
            p = os.path.join(p, 'hole')
        return p

    def annotation_path_i(self, i):
        return os.path.join(self.__repr__(), f'{self.model_category}{i}_dataset.csv')

    def annotations_paths(self):
        return [self.annotation_path_i(i) for i in range(50)]

    def merged_annotation_path(self):
        return os.path.join(self.__repr__(), 'merged_dataset.csv')

    # For each model category:
    # 0 for non-bin unscaled
    # 1 for non-bin scaled
    # 2-4 for each restriction axes scaled
    # 5-7 for each restriction axes
    # 8-9 for bin in holes for axes X,Y
    @classmethod
    def get_dataset(cls, i):
        MODEL_CATEGORY = ['plane', 'car', 'lamp', 'SM'][i // 10]

        CATEGORY_JOB_ID = i % 10

        if CATEGORY_JOB_ID < 2:
            TYPE = DatasetType.Full
        elif CATEGORY_JOB_ID >= 8:
            TYPE = DatasetType.Hole
        else:
            TYPE = DatasetType.Bin

        SCALE = 1 <= CATEGORY_JOB_ID <= 4 or CATEGORY_JOB_ID == 9
        AXES_JOB_ID = CATEGORY_JOB_ID - 2
        RESTRICTION_AXES = [(0, 1), (1, 2), (0, 2)][AXES_JOB_ID % 3 if CATEGORY_JOB_ID < 8 else 0]
        return cls(MODEL_CATEGORY, TYPE, SCALE, RESTRICTION_AXES)


@dataclass
class IDatasetPath:
    model_category: str
    type: DatasetType

    def __post_init__(self):
        if self.type == DatasetType.Hole or self.type == DatasetType.NonBin:
            print(f'Error: incorrect dataset type of {self.type} for IDatasetPath')
        self.path = self.__repr__()
        self.image_path = os.path.join(self.path, 'images')
        self.config_file = os.path.join(self.path, 'configuration.txt')

    def __repr__(self):
        p = os.path.join('/home/avic/om2/datasets/ilab', self.model_category)

        if self.type == DatasetType.Full:
            p = os.path.join(p, 'full')
        elif self.type == DatasetType.Bin:
            p = os.path.join(p, 'bin')
        return p

    def annotation_path_i(self, i):
        return os.path.join(self.__repr__(), f'{self.model_category}{i}_dataset.csv')

    def annotations_paths(self):
        return [self.annotation_path_i(i) for i in range(50)]

    def merged_annotation_path(self):
        return os.path.join(self.__repr__(), 'merged_dataset.csv')

    # For each model category:
    # 0 for non-bin
    # 1 for bin

    @classmethod
    def get_dataset(cls, i):
        MODEL_CATEGORY = ['plane', 'car', 'tank', 'train'][i // 2]
        return cls(MODEL_CATEGORY, DatasetType(i % 2))