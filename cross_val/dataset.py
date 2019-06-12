import glob
import os
import sys
import numpy as np
import cv2
import torch
from albumentations import (
	RandomRotate90, Normalize,
	Flip, OneOf, Compose)
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('..')


# from utils.face_detection.face_detection import get_face

class IDRND_dataset_CV(Dataset):
	def __init__(self, path_to_csv='cross_val_DF.csv', fold=0, mode='train', double_loss_mode=False,
				 add_idrnd_v1_dataset=False, output_shape=300):

		self.cross_val_DF = pd.read_csv(path_to_csv)[['users', f'fold_{fold}']]
		self.users = self.cross_val_DF[self.cross_val_DF[f'fold_{fold}'] == mode]['users'].values
		self.mode = mode
		self.output_shape = output_shape
		self.double_loss_mode = double_loss_mode

		images = [img for user in self.users for img in glob.glob(user + '/*.png')]
		self.masks = [img for img in images if '2dmask' in img]
		self.printed = [img for img in images if 'printed' in img]
		self.replay = [img for img in images if 'replay' in img]
		self.real = [img for img in images if 'real' in img]

		if add_idrnd_v1_dataset:
			self.idrnd_v1_images = glob.glob("../data/idrnd_v1/*/*.png")
			self.idrnd_v1_replay = [i for i in self.idrnd_v1_images if 'real' not in i]
			self.idrnd_v1_real = [i for i in self.idrnd_v1_images if 'real' in i]

			self.real += self.idrnd_v1_real
			self.replay += self.idrnd_v1_replay

		if self.mode == 'train':
			self.aug = self.get_aug()
		else:
			self.aug = self.get_aug(p=0.0)
		self.images = self.masks + self.printed + self.replay + self.real
		if self.double_loss_mode:
			self.labels = [1] * len(self.masks) + [2] * len(self.printed) + [3] * len(self.replay) + [0] * len(
				self.real)
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
			], p=p),
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
		image = self.aug(image=image)['image']  # / 255.
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


if __name__ == '__main__':
	dataset = IDRND_dataset_CV(mode='train', add_idrnd_v1_dataset=True)
	batch = dataset[1]
	user_id = batch['user_id']
	i = batch['image']
	i = i.numpy()
	i = np.moveaxis(i, 0, -1)
	plt.imshow(i)
	plt.show()
