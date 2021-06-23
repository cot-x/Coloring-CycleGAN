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


class Util:
    @staticmethod
    def toLineDrawing(img, iterations=1):
        PIL = transforms.ToPILImage()
        ToTensor = transforms.ToTensor()
        
        img = np.asarray(PIL(img))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 8近傍
        neiborhood8 = np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]],
                                np.uint8)

        # 膨張処理
        img_dilate = cv2.dilate(img, neiborhood8, iterations=iterations)
        # 差分
        img_diff = cv2.absdiff(img, img_dilate)
        # ネガポジ反転
        img_diff_not = cv2.bitwise_not(img_diff)

        img = cv2.cvtColor(img_diff_not, cv2.COLOR_GRAY2RGB)
        img = ToTensor(img)

        return img
    
    class LineDrawing:
        def __init__(self, iterations=1):
            self.iterations = iterations

        def __call__(self, img):
            return toLineDrawing(img, self.iterations)
        
    @staticmethod
    def loadImages(batch_size, folder_path, size):
        imgs = ImageFolder(folder_path, transform=transforms.Compose([
            transforms.Resize(int(size)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
        return DataLoader(imgs, batch_size=batch_size, shuffle=True, drop_last=True)

    @staticmethod
    def showImages(image1, image2):
        #%matplotlib inline
        import matplotlib.pyplot as plt
        
        PIL = transforms.ToPILImage()
        ToTensor = transforms.ToTensor()

        img1 = PIL(image1[0])
        img2 = PIL(image2[0])
        fig = plt.figure(dpi=200)
        ax = fig.add_subplot(1, 2, 1) # (row, col, num)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(img1)
        ax = fig.add_subplot(1, 2, 2) # (row, col, num)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(img2)
        #plt.gray()
        plt.show()
