import sys
from torch import is_tensor, cuda
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, WeightedRandomSampler, RandomSampler, SequentialSampler
from torchvision import transforms
from torchvision.io import read_image
from sklearn.preprocessing import LabelEncoder
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from utils.persistent_data_class import *


def get_base_mask(df, base_orientations):
    return (df.object_x.between(*base_orientations[0])) & (df.object_y.between(*base_orientations[1])) & (df.object_z.between(*base_orientations[2]))


def get_dataloader(dataset, shuffle, batch_size, num_workers=4):
    
    if shuffle:
        dataset_sampler = WeightedRandomSampler(dataset.frame.weights, len(dataset), False)
    else:
        dataset_sampler = SequentialSampler(dataset)
    
    if dataset.preload_dataset:
        
        return DataLoader(dataset=dataset,
                          sampler=BatchSampler(dataset_sampler,
                                               batch_size=batch_size,
                                               drop_last=False))
    else:
        return DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          sampler=dataset_sampler,
                          num_workers=num_workers,
                          pin_memory=True)

@dataclass
class RotationDataset:

    exp_data: ExpData

    def __post_init__(self):
        
        for frame in self.exp_data.frames:
            if frame.df is None:
                self.setup_frames()
                break

        self.instance_codec = preprocessing.LabelEncoder()
        self.instance_codec.fit(self.exp_data.full_instances + self.exp_data.held_instances + self.exp_data.partial_instances)

        group_cache = {}
        self.datasets = [_Dataset(name, self, frame.df, True, self.exp_data.augment, group_cache)
                         for frame, name in zip(self.exp_data.frames, (['training',
                                                                        'full validation',
                                                                        'partial validation',
                                                                        'partial OOD',
                                                                        'partial OOD subset']))]
        
        del group_cache
        
        self.data_loaders = [get_dataloader(dataset, shuffle, self.exp_data.batch_size)
                             for dataset, shuffle in zip(self.datasets, (True, False, False, False, False, False))]
        
        self.train_loader, self.full_validation_loader, self.partial_validation_loader, self.partial_ood_loader, self.partial_ood_subset_loader = self.data_loaders
        self.eval_loaders = self.data_loaders[1:-1]

    def __repr__(self):
        return format_vars(self)

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
        
        full_training_frame_partition = np.random.choice(len(full_instances_frame), 10_000)

        full_training_frame = full_instances_frame[~full_instances_frame.index.isin(full_training_frame_partition)].copy()
        full_training_frame['weights'] = len(self.exp_data.full_instances)
            
        self.exp_data.full_validation_frame.df = full_instances_frame[full_instances_frame.index.isin(full_training_frame_partition)].copy()
        
        if len(self.exp_data.held_instances) > 0:
            held_instances_frame = full_instances_frame[full_instances_frame.instance_name.isin(self.exp_data.held_instances)].copy()
            held_base_mask = get_base_mask(held_instances_frame, self.exp_data.base_orientations)
            held_frame = held_instances_frame[held_base_mask]
            held_frame['weights'] = len(self.exp_data.held_instances)
        

        partial_base_mask = get_base_mask(partial_instances_frame, self.exp_data.base_orientations)

        self.exp_data.partial_validation_frame.df = partial_instances_frame[partial_base_mask].copy()
        self.exp_data.partial_validation_frame.df['weights'] = len(self.exp_data.partial_instances)
        self.exp_data.partial_ood_frame.df = partial_instances_frame[~partial_base_mask].copy()
        self.exp_data.partial_ood_frame_subset.df = self.exp_data.partial_ood_frame.df.sample(n=10_000)
        
        if len(self.exp_data.held_instances) > 0:
            self.exp_data.training_frame.df = pd.concat([full_training_frame, held_frame, self.exp_data.partial_validation_frame.df])
        else:
            self.exp_data.training_frame.df = pd.concat([full_training_frame, self.exp_data.partial_validation_frame.df])
            
        for frame in self.exp_data.frames:
            frame.dump()


@dataclass
class _Dataset(Dataset):
    name: str
    dataset: RotationDataset
    frame: pd.DataFrame
    preload_dataset: bool
    augment: bool
    group_cache: {str : object} = None

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
            self.loaded_dataset = None
            for name, group in tqdm(self.frame.groupby('image_group'), file=sys.stdout):

                group_arr = Arr.from_path(group.image_group.iloc[0])
                if self.group_cache is not None:
                    if group_arr.file_path not in self.group_cache:
                        self.group_cache[group_arr.file_path] = group_arr.arr
                    try:
                        loaded_images = self.group_cache[group_arr.file_path][group.image_idx]
                    except:
                        print('error loading images')
                        print(self.group_cache)
                        print(group_arr.file_path)
                        import pdb; pdb.set_trace()
                    
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
            image = image.unsqueeze(-3)
        
        else:
            idx.tolist() # TODO check if it is necessary to turn to a list in this condition
            entry = self.frame.iloc[idx]
            image = read_image(entry.image_name) / 255
        
        image = transforms.Pad((224 - image.shape[-1]) // 2)(image)
        image = self.affine_transform(image)
        # image = transforms.Normalize(image.mean((-4, -2, -1)), image.std((-4, -2, -1)))(image)
        image = transforms.Normalize(tuple(image.mean((-2, -1))), tuple(image.std((-2, -1))))(image.squeeze(axis=1)).unsqueeze(axis=1)
        assert image.shape[-2] == 224 and image.shape[-1] == 224
        broadcast_shape = np.array(image.shape)
        broadcast_shape[-3] = 3
        return image.broadcast_to(tuple(broadcast_shape))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        
        return (self.get_img(idx),
                (self.dataset.instance_codec.transform(self.frame.iloc[idx].instance_name)))

# def set_augmentation(self, state):
    #     def transforms_seq(img_name):
    #         img = read_image(img_name)
    #         #TODO is this the correct order of transforms? Or should I do it the other way?
    #         if self.exp_data.model_type == ModelType.Inception:
    #             img = transforms.Resize((299, 299), Image.BICUBIC)(img)
    #         elif self.exp_data.model_type == ModelType.Equivariant:
    #             img = transforms.Pad((0, 0, 1, 1), fill=65)(img)
    #         if state:
    #             img = transforms.RandomRotation((-180, 180), resample=Image.BICUBIC, fill=(65, 65, 65))(img)
    #         img = transforms.ToTensor()(img)
    #         return transforms.Normalize(img.mean((1, 2)), img.std((1, 2)))(img)
    #
    #     self.transform_op = transforms_seq
