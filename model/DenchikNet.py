import torch.nn as nn


class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, x):
		return x


class DenchikModel(nn.Module):
	def __init__(self, n_features=32):
		super(DenchikModel, self).__init__()

		# 5 224 224
		self.block1 = nn.Sequential(
			nn.Conv3d(3, n_features, kernel_size=3, stride=1, padding=(1, 1, 1), dilation=2),
			nn.BatchNorm3d(n_features),
			nn.ReLU(inplace=True),
			nn.Conv3d(n_features, n_features, kernel_size=3, stride=1, padding=(1, 0, 0), dilation=1),
			nn.BatchNorm3d(n_features),
			nn.ReLU(inplace=True),
			nn.MaxPool3d(kernel_size=(1, 3, 3))
		)
		# 5 224 224
		self.block2 = nn.Sequential(
			nn.Conv3d(n_features, n_features * 2, kernel_size=3, stride=1, padding=(1, 1, 1), dilation=2),
			nn.BatchNorm3d(n_features * 2),
			nn.ReLU(inplace=True),
			nn.Conv3d(n_features * 2, n_features * 2, kernel_size=3, stride=1, padding=(1, 0, 0), dilation=1),
			nn.BatchNorm3d(n_features * 2),
			nn.ReLU(inplace=True),
			nn.MaxPool3d(kernel_size=(1, 3, 3))
		)

		self.block3 = nn.Sequential(
			nn.Conv3d(n_features * 2, n_features * 4, kernel_size=3, stride=1, padding=(1, 1, 1), dilation=1),
			nn.BatchNorm3d(n_features * 4),
			nn.ReLU(inplace=True),
			nn.Conv3d(n_features * 4, n_features * 4, kernel_size=3, stride=1, padding=(1, 0, 0), dilation=1),
			nn.BatchNorm3d(n_features * 4),
			nn.ReLU(inplace=True),
			nn.MaxPool3d(kernel_size=(1, 3, 3))
		)

		self.block4 = nn.Sequential(
			nn.Conv3d(n_features * 4, n_features * 8, kernel_size=3, stride=1, padding=(1, 1, 1), dilation=1),
			nn.BatchNorm3d(n_features * 8),
			nn.ReLU(inplace=True),
			nn.Conv3d(n_features * 8, n_features * 8, kernel_size=3, stride=1, padding=(1, 0, 0), dilation=1),
			nn.BatchNorm3d(n_features * 8),
			nn.ReLU(inplace=True),
			#nn.MaxPool3d(kernel_size=(1, 3, 3)),
			nn.AvgPool3d(kernel_size=(1, 5, 5), stride=1),
			nn.
			nn.Linear(256, 64),
			nn.ReLU(),
			nn.BatchNorm1d(64),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.BatchNorm1d(32),
			nn.Linear(32, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		return x
