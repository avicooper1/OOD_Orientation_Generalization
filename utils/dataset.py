import sys
from torch import unsqueeze
from torch import cuda
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, WeightedRandomSampler, SequentialSampler
from torchvision import transforms
from torchvision.io import read_image
from sklearn.preprocessing import LabelEncoder
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from utils.persistent_data_class import *
from utils.tools import get_base_mask


def get_dataloader(dataset, shuffle, batch_size):
    
    if shuffle:
        dataset_sampler = WeightedRandomSampler(dataset.frame.weights, len(dataset.frame), True)
    else:
        dataset_sampler = SequentialSampler(dataset)
    
    return DataLoader(dataset=dataset,
                      sampler=BatchSampler(dataset_sampler, batch_size, drop_last=False))


@dataclass
class RotationDataset:

    exp_data: ExpData

    def __post_init__(self):
        
        for frame in self.exp_data.frames:
            if frame.df is None:
                self.setup_frames()
                break

        self.instance_codec = LabelEncoder()
        self.instance_codec.fit(self.exp_data.full_instances + self.exp_data.held_instances + self.exp_data.partial_instances)

        group_cache = {}
        self.datasets = [_Dataset(name, self, frame.df, True, self.exp_data.augment, group_cache)
                         for frame, name in zip(self.exp_data.frames, (['training',
                                                                        'full validation',
                                                                        'partial base',
                                                                        'partial OOD',]))]
        
        del group_cache
        
        self.data_loaders = [get_dataloader(dataset, shuffle, self.exp_data.batch_size)
                             for dataset, shuffle in zip(self.datasets, (True, False, False, False, False, False))]
        
        self.train_loader, self.full_validation_loader, self.partial_base_loader, self.partial_ood_loader = self.data_loaders
        
        # We change the order of the dataloaders to ensure that the partial_base_loader is run first, and if it is at peak
        # we save the results of the other dataloaders
        self.eval_loaders = self.data_loaders[1:]

    def setup_frames(self):

        assert self.exp_data.epochs_completed == 0, 'Attempting to regenerate training and evaluation frames after training'

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
        
        full_training_frame_partition = np.random.choice(len(full_instances_frame), 10_000, replace=False)
        
        self.exp_data.full_validation_frame.df = full_instances_frame.iloc[full_training_frame_partition].copy()

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
            
            self.exp_data.training_frame.df = pd.concat([full_training_frame, held_frame, self.exp_data.partial_base_frame.df])
            
        else:
            self.exp_data.training_frame.df = pd.concat([full_training_frame, self.exp_data.partial_base_frame.df])
            
        for frame in self.exp_data.frames:
            frame.dump()


@dataclass
class _Dataset(Dataset):
    name: str
    dataset: RotationDataset
    frame: pd.DataFrame
    preload_dataset: bool
    augment: bool
    group_cache: {str: object} = None
    loaded_dataset: cuda.ByteTensor = None

    def __post_init__(self):
        self.frame.reset_index(drop=True, inplace=True)

        # Note: During dataset generation / rendering, images are guaranteed to have 20 pixels of padding.
        # Images are 224x224 pixels. 20 / 224 ~ 0.089, which allows for up to this much translation without losing
        # parts of the object
        # TODO We do implement scaling. Are we ok if parts of the image might be lost? Confirm these parameters
        self.affine_transform = transforms.RandomAffine(degrees=(-180, 180) if self.augment else 0,
                                                        translate=(0.08, 0.08),
                                                        scale=(0.8, 1.2),
                                                        interpolation=InterpolationMode.BILINEAR)
        
        if self.preload_dataset:
            for name, group in tqdm(self.frame.groupby('image_group'), file=sys.stdout, desc=f'Loading {self.name} dataset from group file'):

                group_arr = Arr.from_path(group.image_group.iloc[0])
                if self.group_cache is not None:
                    if group_arr.file_path not in self.group_cache:
                        self.group_cache[group_arr.file_path] = group_arr.arr
                    loaded_images = self.group_cache[group_arr.file_path][group.image_idx]
                    
                else:
                    loaded_images = group_arr.arr[group.image_idx]
                
                if self.loaded_dataset is None:
                    self.loaded_dataset = cuda.ByteTensor(len(self.frame), loaded_images.shape[1], loaded_images.shape[1])
                
                self.loaded_dataset[group.index] = cuda.ByteTensor(loaded_images)
            if self.group_cache is not None:
                del self.group_cache

    def get_img(self, idx):
        if self.preload_dataset:
            image = self.loaded_dataset[idx].float()
        
        else:
            idx.tolist() # TODO check if it is necessary to turn to a list in this condition
            entry = self.frame.iloc[idx]
            image = read_image(entry.image_name) / 255

        if type(idx) == int:
            image = unsqueeze(image, 0)

        image = transforms.Pad((224 - image.shape[-1]) // 2)(image)
        # image = self.affine_transform(image)
        # import pdb; pdb.set_trace()
        image = transforms.Normalize(tuple(image.mean((-2, -1))), tuple(image.std((-2, -1))))(image)
        # transforms.Normalize(image.mean(), image.std())(image)
        return image.unsqueeze(axis=1)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.get_img(idx), self.dataset.instance_codec.transform(self.frame.iloc[idx].instance_name)
