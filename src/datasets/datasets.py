import glob
import sys

sys.path.append('..')

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


def filter_paths_by_class(paths, cls):
    return [path for path in paths if cls in path]


class TestDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __getitem__(self, item):
        image_attrs = self.paths[item]
        image = cv2.imread(image_attrs['path'])[:, :, ::-1]

        image = self.transform(image=image)['image']
        image = np.moveaxis(image, -1, 0)
        return image_attrs['id'], image_attrs['frame'], torch.tensor(image, dtype=torch.float)

    def __len__(self):
        return len(self.paths)


class TrainDataset(Dataset):
    def __init__(self, users, transform):
        self.transform = transform

        paths = list()
        for user in users:
            paths.extend(glob.glob(user + '/*.png'))

        masks = filter_paths_by_class(paths, '2dmask')
        printed = filter_paths_by_class(paths, 'printed')
        replay = filter_paths_by_class(paths, 'replay')
        real = filter_paths_by_class(paths, 'real')

        self.paths = np.asarray(masks + printed + replay + real)
        self.labels = np.asarray([1] * len(masks) + [2] * len(printed) + [3] * len(replay) + [0] * len(real))

    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, item):
        path = self.paths[item]
        image = cv2.imread(path)[:, :, ::-1]
        label = self.labels[item]
        image = self.transform(image=image)['image']
        image = np.moveaxis(image, -1, 0)

        return {
            'image': torch.tensor(image, dtype=torch.float),
            'cross_label': torch.tensor(label, dtype=torch.long),
            'bin_label': torch.tensor(1 - int(label == 0), dtype=torch.float),
            'user_id': path.split('/')[-2],
            'frame': path[-6:-4]}
