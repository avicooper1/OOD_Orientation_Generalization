from dataclasses import dataclass, field
import os
from ast import literal_eval
import warnings
import pandas as pd
import numpy as np
import gzip
from abc import ABC, abstractmethod
from pathlib import Path


@dataclass
class PersistentDataObject(ABC):
    dir_path: str
    file_name: str
    extension: str
    eager_dump: bool = False
    overwrite_warn: bool = False
    _data = None

    def __post_init__(self):
        self.file_path = os.path.join(self.dir_path, f'{self.file_name}.{self.extension}')

    @classmethod
    def from_path(cls, path):
        path_as_pathlib = Path(path)
        dir_path = str(path_as_pathlib.parent)
        file_name = path_as_pathlib.stem.split('.')[0]
        extension = ''.join(path_as_pathlib.suffixes)[1:]
        return cls(dir_path, file_name, extension)

    @property
    def data(self):
        if self._data is None:
            self._data = self.load()
        return self._data

    @data.setter
    def data(self, data):
        if self.overwrite_warn and self._data is not None:
            warnings.warn('Overwriting pre-existing data')
        self._data = data

        if self.eager_dump:
            self.dump()

    def reload_data(self):
        self._data = self.load()
        return self._data

    def on_disk(self):
        return os.path.exists(self.file_path)

    @abstractmethod
    def loader(self):
        pass

    @abstractmethod
    def dumper(self):
        pass

    def load(self):
        if self.on_disk():
            return self.loader()

    def dump(self):
        if self._data is not None:
            self.dumper()

    def __repr__(self):
        return self.file_path


@dataclass()
class Arr(PersistentDataObject):

    extension: str = 'npy'
    compress: bool = False

    def __post_init__(self):

        if self.compress:
            self.extension += '.gz'
        else:
            if self.extension[-2:] == 'gz':
                self.compress = True

        super().__post_init__()

    @property
    def arr(self):
        return self.data

    @arr.setter
    def arr(self, arr):
        self.data = arr

    def loader(self):
        if self.compress:
            with gzip.GzipFile(self.file_path, 'r') as f:
                return np.load(f)
        else:
            return np.load(self.file_path)

    def dumper(self):
        if self.compress:
            with gzip.GzipFile(self.file_path, 'w') as f:
                np.save(f, self.arr, allow_pickle=False)
        else:
            np.save(self.file_path, self.arr)


@dataclass
class DF(PersistentDataObject):

    extension: str = 'csv'

    converters = None
    delimiter = ','
    setup = None
    destruct = None

    @property
    def df(self):
        return self.data

    @df.setter
    def df(self, df):
        self.data = df

    def loader(self):
        read_df = pd.read_csv(self.file_path, converters=self.converters, delimiter=self.delimiter)
        if self.setup:
            self.setup(read_df)
        return read_df

    def dumper(self):
        (self.destruct() if self.destruct else self.df).to_csv(self.file_path, index=False, sep=self.delimiter)


def get_path_components(s, num_components):
    return os.path.join(*s.split(os.sep)[-num_components:])


@dataclass
class AnnotationFile(DF):
    converters: dict = field(default_factory=lambda: {"cubelet_i": literal_eval})
    delimiter: str = ';'
    storage_path: str = None

    def setup(self, df):
        assert self.storage_path is not None
        df.cubelet_i = df.cubelet_i.apply(np.array)
        df.image_name = df.image_name.apply(lambda name: os.path.join(self.storage_path, name))
        if 'image_group' in df.columns:
            df.image_group = df.image_group.apply(lambda name: os.path.join(self.storage_path, name))
    
    def destruct(self):
        df_to_write = self.df.copy()
        df_to_write.cubelet_i = df_to_write.cubelet_i.apply(lambda arr: str(tuple(arr)))
        df_to_write.image_name = df_to_write.image_name.apply(lambda name: get_path_components(name, 5))
        if 'image_group' in df_to_write.columns:
            df_to_write.image_group = df_to_write.image_group.apply(lambda group: get_path_components(group, 4))
        return df_to_write
        