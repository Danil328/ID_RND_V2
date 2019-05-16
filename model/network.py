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
		self.base_model.fc = Identity()

		self.classifier = nn.Sequential(
			nn.Linear(in_features = 512, out_features = 128),
			nn.ELU(),
			nn.BatchNorm1d(num_features = 128),
			nn.Linear(in_features = 128, out_features = 32),
			nn.ELU(),
			nn.BatchNorm1d(num_features = 32),
			nn.Linear(in_features = 32, out_features = 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.base_model(x)
		x = self.classifier(x)
		return x