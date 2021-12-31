import torch
from PIL import Image
from sklearn import preprocessing
from torchvision import transforms
import numpy as np

import sys
sys.path.append('/home/avic/Rotation-Generalization')
from my_dataclasses import *

class RotationDataset:

    def __init__(self, exp_data: ExpData):

        self.exp_data = exp_data

        self.training_category = exp_data.training_category
        self.testing_category = exp_data.testing_category

        self.training = Ann(exp_data, True)
        self.testing = Ann(exp_data, False, hole=self.exp_data.hole == 1)
        if self.exp_data.hole == 2:
            self.holing = Ann(exp_data, False, hole=True)

        non_testing_indices = np.r_[:40]
        self.training_models = self.training.full.models[non_testing_indices[:exp_data.data_div]]
        self.held_models = self.training.bin.models[non_testing_indices[exp_data.data_div:]]
        self.testing_models = self.testing.bin.models[40:]

        self.exp_models = np.concatenate([self.training_models, self.held_models, self.testing_models])
        self.obj_cat_codec = preprocessing.LabelEncoder()
        self.obj_cat_codec.fit(self.exp_models)

        self.set_augmentation(False)

        self.splits = self._get_splits()

    def set_augmentation(self, state):
        def transforms_seq(img_name):
            img = Image.open(img_name).convert('RGB')
            #TODO is this the correct order of transforms? Or should I do it the other way?
            if self.exp_data.model_type == ModelType.Inception:
                img = transforms.Resize((299, 299), Image.BICUBIC)(img)
            elif self.exp_data.model_type == ModelType.Equivariant:
                img = transforms.Pad((0, 0, 1, 1), fill=65)(img)
            if state:
                img = transforms.RandomRotation((-180, 180), resample=Image.BICUBIC, fill=(65, 65, 65))(img)
            img = transforms.ToTensor()(img)
            return transforms.Normalize(img.mean((1, 2)), img.std((1, 2)))(img)

        self.transform_op = transforms_seq

    class _Dataset(Dataset):
        def __init__(self, dataset, frame, save_frame=False):
            self.dataset = dataset
            self.frame = frame
            self.frame.reset_index(drop=True, inplace=True)
            if save_frame:
                self.frame.to_csv(self.dataset.exp_data.testing_frame_path)

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            entry = self.frame.iloc[idx]
            return self.dataset.transform_op(entry.image_name), self.dataset.obj_cat_codec.transform([entry.model_name])

    def _get_splits(self):

        def dataloader(frame, shuffle=False, sample=False, save_frame=False):
            if len(frame) == 0:
                return None
            d = self._Dataset(self, frame.sample(n=2_000) if sample else frame, save_frame=save_frame)
            return DataLoader(d, batch_size=self.exp_data.batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)

        if self.exp_data.hole == 2:
            training_from_testing = pd.concat([
                    self.testing.bin.ann[self.testing.bin.ann.model_name.isin(self.testing_models)].sample(n=2_000),
                    self.holing.bin.ann[self.holing.bin.ann.model_name.isin(self.testing_models)].sample(n=2_000)])
        else:
            training_from_testing = self.testing.bin.ann[self.testing.bin.ann.model_name.isin(self.testing_models)]

        training_frame = pd.concat([
            self.training.full.ann[self.training.full.ann.model_name.isin(self.training_models)].sample(frac=0.5),
            self.training.bin.ann[self.training.bin.ann.model_name.isin(self.held_models)].sample(frac=0.5),
            training_from_testing.sample(frac=0.5)])

        training_models_validation_frame = self.training.bin.ann[self.training.bin.ann.model_name.isin(self.training_models)]
        held_models_validation_frame = self.training.full.ann[self.training.full.ann.model_name.isin(self.held_models)]

        testing_frame = self.testing.full.ann[self.testing.full.ann.model_name.isin(self.testing_models)]

        return Splits(dataloader(training_frame, shuffle=True), dataloader(testing_frame, save_frame=True), dataloader(training_models_validation_frame, sample=True), dataloader(held_models_validation_frame, sample=True))


    def __repr__(self):
        return format_vars(self)
