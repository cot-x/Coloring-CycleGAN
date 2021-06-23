import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image

from tqdm import tqdm
from PIL import Image, ImageFile
from pickle import load, dump
import os
import cv2
import itertools
import argparse

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FReLU(nn.Module):
    def __init__(self, n_channel, kernel=3, stride=1, padding=1):
        super().__init__()
        self.funnel_condition = nn.Conv2d(n_channel, n_channel, kernel_size=kernel,stride=stride, padding=padding, groups=n_channel)
        self.bn = nn.BatchNorm2d(n_channel)

    def forward(self, x):
        tx = self.bn(self.funnel_condition(x))
        out = torch.max(x, tx)
        return out

class ResidualSEBlock(nn.Module):
    def __init__(self, in_features, reduction=16):
        super().__init__()

        self.shortcut = nn.Sequential()
        self.residual = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=0)),
            nn.InstanceNorm2d(in_features),
            FReLU(in_features),
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=0)),
            nn.InstanceNorm2d(in_features)
        )

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_features, in_features // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // reduction, in_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)
        squeeze = self.squeeze(residual)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze).view(residual.size(0), residual.size(1), 1, 1)
        return F.relu(residual * excitation.expand_as(residual) + shortcut)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=8):
        super().__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            FReLU(64)
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.utils.spectral_norm(nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1)),
                nn.InstanceNorm2d(out_features),
                FReLU(out_features)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ ResidualSEBlock(in_features) ]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.utils.spectral_norm(nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1)),
                nn.InstanceNorm2d(out_features),
                FReLU(out_features)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7),
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super().__init__()

        model = [
            nn.utils.spectral_norm(nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(128), 
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(256), 
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=4, padding=1)),
            nn.InstanceNorm2d(512), 
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # Fully Convolutional Network
        model += [ nn.Conv2d(512, 1, kernel_size=4, padding=1) ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.shape[0], -1) # Global Average Pooling
        return x
