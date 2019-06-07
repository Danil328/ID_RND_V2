from __future__ import absolute_import

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.nn import init


class ConvBlock(nn.Module):
	"""Basic convolutional block:
	convolution + batch normalization + relu.
	Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
	- in_c (int): number of input channels.
	- out_c (int): number of output channels.
	- k (int or tuple): kernel size.
	- s (int or tuple): stride.
	- p (int or tuple): padding.
	"""

	def __init__(self, in_c, out_c, k, s=1, p=0):
		super(ConvBlock, self).__init__()
		self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
		self.bn = nn.BatchNorm2d(out_c)

	def forward(self, x):
		return F.relu(self.bn(self.conv(x)))


class SpatialAttn(nn.Module):
	"""Spatial Attention Layer"""

	def __init__(self):
		super(SpatialAttn, self).__init__()

	def forward(self, x):
		# global cross-channel averaging # e.g. 32,2048,24,8
		x = x.mean(1, keepdim=True)  # e.g. 32,1,24,8
		h = x.size(2)
		w = x.size(3)
		x = x.view(x.size(0), -1)  # e.g. 32,192
		z = x
		for b in range(x.size(0)):
			z[b] /= torch.sum(z[b])
		z = z.view(x.size(0), 1, h, w)
		return z


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
		   'resnet152']


class ResNet(nn.Module):
	__factory = {
		18: torchvision.models.resnet18,
		34: torchvision.models.resnet34,
		50: torchvision.models.resnet50,
		101: torchvision.models.resnet101,
		152: torchvision.models.resnet152,
	}

	def __init__(self, depth=50, pretrained=True, num_features=512, dropout=0, num_classes=10):
		super(ResNet, self).__init__()

		self.depth = depth
		self.pretrained = pretrained

		# Construct base (pretrained) resnet
		if depth not in ResNet.__factory:
			raise KeyError("Unsupported depth:", depth)
		self.base = ResNet.__factory[depth](pretrained=pretrained)

		for mo in self.base.layer4[0].modules():
			if isinstance(mo, nn.Conv2d):
				mo.stride = (1, 1)

		self.num_features = num_features
		self.num_classes = num_classes  # num_classes to be changed according to dataset
		self.dropout = dropout
		out_planes = self.base.fc.in_features
		self.local_conv = nn.Conv2d(out_planes, self.num_features, kernel_size=1, padding=0, bias=False)
		self.local_conv_layer1 = nn.Conv2d(256, self.num_features, kernel_size=1, padding=0, bias=False)
		self.local_conv_layer2 = nn.Conv2d(512, self.num_features, kernel_size=1, padding=0, bias=False)
		self.local_conv_layer3 = nn.Conv2d(1024, self.num_features, kernel_size=1, padding=0, bias=False)

		init.kaiming_normal(self.local_conv.weight, mode='fan_out')
		#       init.constant(self.local_conv.bias,0)
		self.feat_bn2d = nn.BatchNorm2d(self.num_features)  # may not be used, not working on caffe
		init.constant(self.feat_bn2d.weight, 1)  # initialize BN, may not be used
		init.constant(self.feat_bn2d.bias, 0)  # iniitialize BN, may not be used

		self.SA1 = SpatialAttn()
		self.SA2 = SpatialAttn()
		self.SA3 = SpatialAttn()
		self.SA4 = SpatialAttn()

		# self.offset = ConvOffset2D(256)

		##---------------------------stripe1----------------------------------------------#
		self.instance0 = nn.Linear(self.num_features, self.num_classes)
		init.normal(self.instance0.weight, std=0.001)
		init.constant(self.instance0.bias, 0)
		##---------------------------stripe1----------------------------------------------#
		##---------------------------stripe1----------------------------------------------#
		self.instance1 = nn.Linear(self.num_features, self.num_classes)
		init.normal(self.instance1.weight, std=0.001)
		init.constant(self.instance1.bias, 0)
		##---------------------------stripe1----------------------------------------------#
		##---------------------------stripe1----------------------------------------------#
		self.instance2 = nn.Linear(self.num_features, self.num_classes)
		init.normal(self.instance2.weight, std=0.001)
		init.constant(self.instance2.bias, 0)
		##---------------------------stripe1----------------------------------------------#
		##---------------------------stripe1----------------------------------------------#
		self.instance3 = nn.Linear(self.num_features, self.num_classes)
		init.normal(self.instance3.weight, std=0.001)
		init.constant(self.instance3.bias, 0)
		##---------------------------stripe1----------------------------------------------#
		##---------------------------stripe1----------------------------------------------#
		self.instance4 = nn.Linear(self.num_features, self.num_classes)
		init.normal(self.instance4.weight, std=0.001)
		init.constant(self.instance4.bias, 0)
		##---------------------------stripe1----------------------------------------------#
		##---------------------------stripe1----------------------------------------------#
		self.instance_layer1 = nn.Linear(self.num_features, self.num_classes)
		init.normal(self.instance_layer1.weight, std=0.001)
		init.constant(self.instance_layer1.bias, 0)
		##---------------------------stripe1----------------------------------------------#
		##---------------------------stripe1----------------------------------------------#
		self.instance_layer2 = nn.Linear(self.num_features, self.num_classes)
		init.normal(self.instance_layer2.weight, std=0.001)
		init.constant(self.instance_layer2.bias, 0)
		##---------------------------stripe1----------------------------------------------#
		##---------------------------stripe1----------------------------------------------#
		self.instance_layer3 = nn.Linear(self.num_features, self.num_classes)
		init.normal(self.instance_layer3.weight, std=0.001)
		init.constant(self.instance_layer3.bias, 0)

		self.fusion_conv = nn.Conv1d(4, 1, kernel_size=1, bias=False)

		self.drop = nn.Dropout(self.dropout)

		if not self.pretrained:
			self.reset_params()

	def forward(self, x):
		for name, module in self.base._modules.items():
			if name == 'avgpool':
				break
			x = module(x)
			if name == 'layer1':
				x_layer1 = x
				continue
			if name == 'layer2':
				x_layer2 = x
			if name == 'layer3':
				x_layer3 = x
		# Add parameter-free spatial attention before GAP
		x_attn1 = self.SA1(x_layer1)
		x_attn2 = self.SA2(x_layer2)
		x_attn3 = self.SA3(x_layer3)

		x_layer1 = x_layer1 * x_attn1
		x_layer2 = x_layer2 * x_attn2
		x_layer3 = x_layer3 * x_attn3
		# Deep Supervision
		x_layer1 = F.avg_pool2d(x_layer1, kernel_size=(56, 56), stride=(1, 1))
		x_layer1 = self.local_conv_layer1(x_layer1)
		x_layer1 = x_layer1.contiguous().view(x_layer1.size(0), -1)
		x_layer1 = self.instance_layer1(x_layer1)

		x_layer2 = F.avg_pool2d(x_layer2, kernel_size=(28, 28), stride=(1, 1))
		x_layer2 = self.local_conv_layer2(x_layer2)
		x_layer2 = x_layer2.contiguous().view(x_layer2.size(0), -1)
		x_layer2 = self.instance_layer2(x_layer2)

		x_layer3 = F.avg_pool2d(x_layer3, kernel_size=(14, 14), stride=(1, 1))
		x_layer3 = self.local_conv_layer3(x_layer3)
		x_layer3 = x_layer3.contiguous().view(x_layer3.size(0), -1)
		x_layer3 = self.instance_layer3(x_layer3)
		# Part-Level Feature
		x = F.avg_pool2d(x, kernel_size=(7, 14), stride=(2, 2))  # H4 W8

		out0 = x / x.norm(2, 1).unsqueeze(1).expand_as(x)  # use this feature vector to do distance measure
		# out0 = torch.cat([f3,out0],dim=1)
		x = self.drop(x)
		x = self.local_conv(x)
		x = self.feat_bn2d(x)
		x = F.relu(x)  # relu for local_conv feature
		x4 = F.avg_pool2d(x, kernel_size=(4, 1), stride=(1, 1))
		x4 = x4.contiguous().view(x4.size(0), -1)

		c4 = self.instance4(x4)

		x = x.chunk(4, 2)
		x0 = x[0].contiguous().view(x[0].size(0), -1)
		x1 = x[1].contiguous().view(x[1].size(0), -1)
		x2 = x[2].contiguous().view(x[2].size(0), -1)
		x3 = x[3].contiguous().view(x[3].size(0), -1)

		c0 = self.instance0(x0)
		c1 = self.instance1(x1)
		c2 = self.instance2(x2)
		c3 = self.instance3(x3)
		return out0, (c0, c1, c2, c3, c4, x_layer1, x_layer2, x_layer3)

	def reset_params(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init.kaiming_normal(m.weight, mode='fan_out')
				if m.bias is not None:
					init.constant(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				init.constant(m.weight, 1)
				init.constant(m.bias, 0)
			elif isinstance(m, nn.Linear):
				init.normal(m.weight, std=0.001)
				if m.bias is not None:
					init.constant(m.bias, 0)


def resnet18(**kwargs):
	return ResNet(18, **kwargs)


def resnet34(**kwargs):
	return ResNet(34, **kwargs)


def resnet50(**kwargs):
	return ResNet(50, **kwargs)


def resnet101(**kwargs):
	return ResNet(101, **kwargs)


def resnet152(**kwargs):
	return ResNet(152, **kwargs)
