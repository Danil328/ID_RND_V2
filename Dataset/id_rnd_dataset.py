import glob
import os
import sys
import numpy as np
import cv2
import torch
from albumentations import (
	RandomRotate90, Normalize,
	Flip, OneOf, Compose,
	ShiftScaleRotate)
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

sys.path.append('..')


class IDRND_dataset(Dataset):
	def __init__(self, path='../data', mode='train', double_loss_mode=False, add_idrnd_v1_dataset=False,
				 use_face_detection=False, output_shape=224):
		self.path_to_data = path
		self.mode = mode
		self.output_shape = output_shape
		self.double_loss_mode = double_loss_mode
		self.use_face_detection = use_face_detection
		self.masks = glob.glob(os.path.join(self.path_to_data, mode, '2dmask/*/*.png'))
		self.printed = glob.glob(os.path.join(self.path_to_data, mode, 'printed/*/*.png'))
		self.replay = glob.glob(os.path.join(self.path_to_data, mode, 'replay/*/*.png'))
		self.real = glob.glob(os.path.join(self.path_to_data, mode, 'real/*/*.png'))

		if add_idrnd_v1_dataset:
			self.idrnd_v1_images = glob.glob("../data/idrnd_v1/*/*.png")
			self.idrnd_v1_replay = [i for i in self.idrnd_v1_images if 'real' not in i]
			self.idrnd_v1_real = [i for i in self.idrnd_v1_images if 'real' in i]

			self.real += self.idrnd_v1_real
			#self.replay += self.idrnd_v1_replay

		if self.mode == 'train':
			self.aug = self.get_aug()
		else:
			self.aug = self.get_aug(p=0.0)
		self.images = self.masks + self.printed + self.replay + self.real
		if self.double_loss_mode:
			self.labels = [1] * len(self.masks) + [2] * len(self.printed) + [3] * len(self.replay) + [0] * len(self.real)
		else:
			self.labels = [1] * len(self.masks + self.printed + self.replay) + [0] * len(self.real)

		self.images = np.asarray(self.images)
		self.labels = np.asarray(self.labels)

		self.count_data()

	def count_data(self):
		print(f'Mask images - {len(self.masks)}\n' +
			  f'Printed images - {len(self.printed)}\n' +
			  f'Replay images - {len(self.replay)}\n' +
			  f'Real images - {len(self.real)}')

		print(f'\nSpoof images - {self.labels.sum()}\n' +
			  f'Real images - {self.__len__() - self.labels.sum()}')

	@staticmethod
	def get_aug(p=.9):
		return Compose([
			OneOf([
				RandomRotate90(),
				Flip(),
				# ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10)
			], p=p),
			# OneOf([
			#     IAAAdditiveGaussianNoise(),
			#     GaussNoise(),
			# ], p = 0.0),
			# OneOf([
			#     MotionBlur(p = .2),
			#     MedianBlur(blur_limit = 3, p = 0.1),
			#     Blur(blur_limit = 3, p = 0.0),
			# ], p = 0.1),
			# OneOf([
			#     OpticalDistortion(p = 0.1),
			#     GridDistortion(p = .1),
			#     IAAPiecewiseAffine(p = 0.3),
			# ], p = 0.0),
			# OneOf([
			#     CLAHE(clip_limit = 2),
			#     IAASharpen(),
			#     IAAEmboss(),
			#     RandomBrightnessContrast(),
			# ], p = 0.0),
			Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
		], p=1.0)

	def __len__(self):
		return self.labels.shape[0]

	def __getitem__(self, idx):
		path_to_image = self.images[idx]
		user_id = path_to_image.split('/')[-2]
		frame = path_to_image[-6:-4]
		image = cv2.imread(path_to_image)
		label = self.labels[idx]

		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		image = cv2.resize(image, (self.output_shape, self.output_shape))
		image = self.aug(image=image)['image'] #/ 255.
		image = np.moveaxis(image, -1, 0)
		if self.double_loss_mode:
			return {"image": torch.tensor(image, dtype=torch.float),
					"label0": torch.tensor(label, dtype=torch.long),
					"label1": torch.tensor(1 - int(label == 0), dtype=torch.float),
					"user_id": user_id,
					"frame": frame}
		return {"image": torch.tensor(image, dtype=torch.float),
				"label": torch.tensor(label, dtype=torch.float),
				"user_id": user_id,
				"frame": frame}


class TestAntispoofDataset(Dataset):
	def __init__(self, paths, output_shape=224):
		self.paths = paths
		self.output_shape = output_shape
		self.aug = self.get_aug()

	def __getitem__(self, index):
		image_info = self.paths[index]
		img = cv2.imread(image_info['path'])

		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (self.output_shape, self.output_shape)) #/ 255.
		img = self.aug(image=img)['image']
		img = np.moveaxis(img, -1, 0)
		return image_info['id'], image_info['frame'], torch.tensor(img, dtype=torch.float)

	def __len__(self):
		return len(self.paths)

	@staticmethod
	def get_aug():
		return Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def make_weights_for_balanced_classes(dataset: Dataset):
	weight_per_class = dict()
	N = dataset.__len__()
	for i in range(len(np.unique(dataset.labels))):
		cnt_element_in_class = len([j for j in dataset.labels if i == j])
		weight_per_class[i] = cnt_element_in_class / N

	weight = list(map(lambda x: weight_per_class[x], dataset.labels))
	return weight


if __name__ == '__main__':
	dataset = IDRND_dataset(mode='train', add_idrnd_v1_dataset=True)
	batch = dataset[1]
	user_id = batch['user_id']

	i = batch['image']
	i = i.numpy()
	i = np.moveaxis(i, 0, -1)
	plt.imshow(i)
	plt.show()
