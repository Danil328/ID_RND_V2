import glob
import os
import sys
import numpy as np
import cv2
import torch
from albumentations import (
    CLAHE, RandomRotate90,
    ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur,
    MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

sys.path.append('..')


class IDRND_dataset(Dataset):
    def __init__(self, path='../data/train', mode='train'):
        self.path_to_data = path
        self.mode = mode
        self.masks = glob.glob(os.path.join(self.path_to_data, '2dmask/*/*01.png'))
        self.printed = glob.glob(os.path.join(self.path_to_data, 'printed/*/*01.png'))
        self.replay = glob.glob(os.path.join(self.path_to_data, 'replay/*/*01.png'))
        self.real = glob.glob(os.path.join(self.path_to_data, 'real/*/*01.png'))

        self.count_data()
        self.aug = self.get_aug()
        self.images = self.masks + self.printed + self.replay + self.real
        self.labels = [1] * len(self.masks + self.printed + self.replay) + [0] * len(self.real)

        self.train_images, self.val_images, self.train_labels, self.val_labels = \
            train_test_split(self.images, self.labels, test_size = 0.2, shuffle = True, random_state = 17)

    def count_data(self):
        if self.mode == 'train':
            print(f'Mask images - {len(self.masks)}\n' +
                  f'Printed images - {len(self.printed)}\n' +
                  f'Replay images - {len(self.replay)}\n' +
                  f'Real images - {len(self.real)}')

    def get_aug(self, p=.5):
        return Compose([
            OneOf([
                RandomRotate90(),
                Flip()
            ]),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p = 0.1),
            OneOf([
                MotionBlur(p = .2),
                MedianBlur(blur_limit = 3, p = 0.1),
                Blur(blur_limit = 3, p = 0.1),
            ], p = 0.1),
            ShiftScaleRotate(shift_limit = 0.0625, scale_limit = 0.1, rotate_limit = 45, p = 0.1),
            OneOf([
                OpticalDistortion(p = 0.1),
                GridDistortion(p = .1),
                IAAPiecewiseAffine(p = 0.3),
            ], p = 0.1),
            OneOf([
                CLAHE(clip_limit = 2),
                IAASharpen(),
                IAAEmboss(),
                RandomBrightnessContrast(),
            ], p = 0.1)
        ], p = p)

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_images)
        else:
            return len(self.val_images)

    def __getitem__(self, idx):
        if self.mode == 'train':
            image = cv2.imread(self.train_images[idx])
            label = self.train_labels[idx]
        else:
            image = cv2.imread(self.val_images[idx])
            label = self.val_labels[idx]

        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.aug(image = image)['image'] / 255.
        image = np.moveaxis(image, -1, 0)
        return {"image": torch.tensor(image, dtype = torch.float), "label": torch.tensor(label, dtype = torch.float)}


class IDRND_test_dataset(Dataset):
    def __init__(self, path='../data/train', mode='train'):
        self.path_to_data = path
        self.mode = mode
        self.masks = glob.glob(os.path.join(self.path_to_data, '2dmask/*/*01.png'))
        self.printed = glob.glob(os.path.join(self.path_to_data, 'printed/*/*01.png'))
        self.replay = glob.glob(os.path.join(self.path_to_data, 'replay/*/*01.png'))
        self.real = glob.glob(os.path.join(self.path_to_data, 'real/*/*01.png'))

        self.count_data()
        self.aug = self.get_aug()
        self.images = self.masks + self.printed + self.replay + self.real
        self.labels = [1] * len(self.masks + self.printed + self.replay) + [0] * len(self.real)

        self.train_images, self.val_images, self.train_labels, self.val_labels = \
            train_test_split(self.images, self.labels, test_size = 0.2, shuffle = True, random_state = 17)

    def count_data(self):
        if self.mode == 'train':
            print(f'Mask images - {len(self.masks)}\n' +
                  f'Printed images - {len(self.printed)}\n' +
                  f'Replay images - {len(self.replay)}\n' +
                  f'Real images - {len(self.real)}')

    def get_aug(self, p=.5):
        return Compose([
            OneOf([
                RandomRotate90(),
                Flip()
            ]),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p = 0.1),
            OneOf([
                MotionBlur(p = .2),
                MedianBlur(blur_limit = 3, p = 0.1),
                Blur(blur_limit = 3, p = 0.1),
            ], p = 0.1),
            ShiftScaleRotate(shift_limit = 0.0625, scale_limit = 0.1, rotate_limit = 45, p = 0.1),
            OneOf([
                OpticalDistortion(p = 0.1),
                GridDistortion(p = .1),
                IAAPiecewiseAffine(p = 0.3),
            ], p = 0.1),
            OneOf([
                CLAHE(clip_limit = 2),
                IAASharpen(),
                IAAEmboss(),
                RandomBrightnessContrast(),
            ], p = 0.1)
        ], p = p)

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_images)
        else:
            return len(self.val_images)

    def __getitem__(self, idx):
        if self.mode == 'train':
            image = cv2.imread(self.train_images[idx])
            label = self.train_labels[idx]
        else:
            image = cv2.imread(self.val_images[idx])
            label = self.val_labels[idx]

        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.aug(image = image)['image'] / 255.
        image = np.moveaxis(image, -1, 0)
        return {"image": torch.tensor(image, dtype = torch.float), "label": torch.tensor(label, dtype = torch.float)}


if __name__ == '__main__':
    dataset = IDRND_dataset()
    batch = dataset.__getitem__(1)
