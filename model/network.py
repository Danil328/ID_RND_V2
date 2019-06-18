import torch
import torch.nn as nn
from torchvision.models import resnet34


class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, x):
		return x


class Model(nn.Module):
	def __init__(self, base_model=resnet34(pretrained=False)):
		super(Model, self).__init__()
		self.base_model = base_model
		self.base_model._fc = Identity()
		self.linear1 = nn.Linear(in_features=1536, out_features=4)
		self.linear2 = nn.Linear(in_features=4, out_features=1)

		self.classifier = nn.Sequential(
			self.base_model,
			nn.Linear(in_features=1536, out_features=128),
			nn.ELU(),
			nn.BatchNorm1d(num_features=128),
			nn.Linear(in_features=128, out_features=1),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.classifier(x)
		x = self.base_model(x)
		x = self.linear1(x)
		x = torch.relu(x)

		return x


class DoubleLossModel(nn.Module):
	def __init__(self, base_model=resnet34(pretrained=False)):
		super(DoubleLossModel, self).__init__()
		self.base_model = base_model
		# self.base_model._fc = Identity()

		self.block1 = nn.Sequential(
			self.base_model,
			nn.Dropout(),
			nn.Linear(in_features=1000, out_features=4),
		)
		self.soft = nn.Softmax()
		self.block2 = nn.Sequential(
			nn.ReLU(),
			nn.Linear(in_features=4, out_features=1),
			nn.Sigmoid()
		)

	def forward(self, x):
		out0 = self.block1(x)
		out1 = self.block2(out0)
		return self.soft(out0), out1


class DoubleLossModelTwoHead(nn.Module):
	def __init__(self, base_model=resnet34(pretrained=False)):
		super(DoubleLossModelTwoHead, self).__init__()
		self.base_model = base_model
		# self.base_model._fc = Identity()

		self.block1 = nn.Sequential(
			nn.Dropout(p=0.25),
			nn.Linear(in_features=1000, out_features=4),
			#nn.Softmax()
		)
		self.block2 = nn.Sequential(
			nn.Dropout(p=0.25),
			nn.Linear(in_features=1000, out_features=1),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.base_model(x)
		out0 = self.block1(x)
		out1 = self.block2(x)
		return out0, out1


if __name__ == '__main__':
	model = DoubleLossModelTwoHead()
