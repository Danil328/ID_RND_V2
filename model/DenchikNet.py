import torch.nn as nn
import torch

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class DenchikModel(nn.Module):
    def __init__(self, n_features=32):
        super(DenchikModel, self).__init__()

        # 5 224 224
        self.block1 = nn.Sequential(
            nn.Conv3d(3, n_features, kernel_size=3, stride=1, padding=(2, 1, 1), dilation=2),
            nn.BatchNorm3d(n_features),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_features, n_features, kernel_size=3, stride=1, padding=(1, 0, 0), dilation=1),
            nn.BatchNorm3d(n_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3))
        )
        # 1 71 71
        self.block2 = nn.Sequential(
            nn.Conv3d(n_features, n_features * 2, kernel_size=3, stride=1, padding=(2, 1, 1), dilation=2),
            nn.BatchNorm3d(n_features * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_features * 2, n_features * 2, kernel_size=3, stride=1, padding=(0, 0, 0), dilation=1),
            nn.BatchNorm3d(n_features * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3))
        )
        # 1 23 23
        self.block3 = nn.Sequential(
            nn.Conv3d(n_features * 2, n_features * 4, kernel_size=3, stride=1, padding=(1, 1, 1), dilation=1),
            nn.BatchNorm3d(n_features * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_features * 4, n_features * 4, kernel_size=3, stride=1, padding=(1, 0, 0), dilation=1),
            nn.BatchNorm3d(n_features * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3))
        )
        # 1 7 7
        self.block4 = nn.Sequential(
            nn.Conv3d(n_features * 4, n_features * 8, kernel_size=3, stride=1, padding=(1, 1, 1), dilation=1),
            nn.BatchNorm3d(n_features * 8),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_features * 8, n_features * 8, kernel_size=3, stride=1, padding=(0, 0, 0), dilation=1),
            nn.BatchNorm3d(n_features * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 5, 5)),
			Flatten())
        # 1 5 5
        self.classifier1 = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4),
            nn.Softmax()
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(256 * 5 * 5, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x4 = self.classifier1(x)
        x = self.classifier2(x)
        return x4, x
