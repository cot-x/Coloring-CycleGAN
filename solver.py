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

from util import *
from model import *


class Solver:
    def __init__(self, use_cpu, lr, num_epochs, batch_size, img_dir, image_size, lambda_cycle, lambda_identity, result_dir, weight_dir):
        use_cuda = torch.cuda.is_available() if not use_cpu else False
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.itype = torch.cuda.LongTensor if use_cuda else torch.LongTensor
        torch.backends.cudnn.benchmark = True
        print(f'Use Device: {self.device}')
        
        self.num_channel = 3
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.result_dir = result_dir
        self.weight_dir = weight_dir
        
        self.dataloader = Util.loadImages(self.batch_size, img_dir, image_size)
        
        self.netG_A2B = Generator(self.num_channel, self.num_channel).to(self.device)
        self.netG_B2A = Generator(self.num_channel, self.num_channel).to(self.device)
        self.netD_A = Discriminator(self.num_channel).to(self.device)
        self.netD_B = Discriminator(self.num_channel).to(self.device)
        self.state_loaded = False

        self.netG_A2B.apply(self.weights_init)
        self.netG_B2A.apply(self.weights_init)
        self.netD_A.apply(self.weights_init)
        self.netD_B.apply(self.weights_init)

        self.optimizer_G = optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()), lr=self.lr, betas=(0, 0.9))
        self.optimizer_D_A = optim.Adam(self.netD_A.parameters(), lr=self.lr * 4, betas=(0, 0.9))
        self.optimizer_D_B = optim.Adam(self.netD_B.parameters(), lr=self.lr * 4, betas=(0, 0.9))
        
        self.last_epoch = 0
        
    def weights_init(self, module):
        if type(module) == nn.Conv2d or type(module) == nn.ConvTranspose2d or type(module) == nn.Linear:
            nn.init.kaiming_normal_(module.weight)
            module.bias.data.fill_(0)
            
    def save_state(self, num):
        self.netG_A2B.cpu()
        self.netG_B2A.cpu()
        self.netD_A.cpu()
        self.netD_B.cpu()
        torch.save(self.netG_A2B.state_dict(), os.path.join(self.weight_dir, f'weight_G_A2B.{num}.pth'))
        torch.save(self.netG_B2A.state_dict(), os.path.join(self.weight_dir, f'weight_G_B2A.{num}.pth'))
        torch.save(self.netD_A.state_dict(), os.path.join(self.weight_dir, f'weight_D_A.{num}.pth'))
        torch.save(self.netD_B.state_dict(), os.path.join(self.weight_dir, f'weight_D_B.{num}.pth'))
        self.netG_A2B.to(self.device)
        self.netG_B2A.to(self.device)
        self.netD_A.to(self.device)
        self.netD_B.to(self.device)
            
    def load_state(self):
        if (os.path.exists('weight_G_A2B.pth') and os.path.exists('weight_G_B2A.pth') and os.path.exists('weight_D_A.pth') and os.path.exists('weight_D_B.pth')):
            self.netG_A2B.load_state_dict(torch.load('weight_G_A2B.pth', map_location=self.device))
            self.netG_B2A.load_state_dict(torch.load('weight_G_B2A.pth', map_location=self.device))
            self.netD_A.load_state_dict(torch.load('weight_D_A.pth', map_location=self.device))
            self.netD_B.load_state_dict(torch.load('weight_D_B.pth', map_location=self.device))
            self.state_loaded = True
            print('Loaded network state.')
    
    def save_resume(self):
        with open(os.path.join('.', 'resume.pkl'), 'wb') as f:
            dump(self, f)
    
    def load_resume(self):
        if os.path.exists('resume.pkl'):
            with open(os.path.join('.', 'resume.pkl'), 'rb') as f:
                print('Load resume.')
                return load(f)
        else:
            return self
    
    def trainGAN(self, epoch, iters, max_iters, real_A, real_B, lambda_cycle=10, lambda_identity=5, a=0, b=1, c=1):
        ### Train CycleGAN with LSGAN.
        ### for example, (a, b, c) = 0, 1, 1 or (a, b, c) = -1, 1, 0
        
        criterion_cycle = nn.L1Loss()
        criterion_identity = nn.L1Loss()

        # ================================================================================ #
        #                               Train the generator                                #
        # ================================================================================ #
        
        # Adversarial Loss
        fake_B = self.netG_A2B(real_A)
        pred_fake = self.netD_B(fake_B)
        loss_GAN_A2B = torch.sum((pred_fake - c) ** 2)
        loss_GAN_A2B = 0.5 * loss_GAN_A2B / self.batch_size

        fake_A = self.netG_B2A(real_B)
        pred_fake = self.netD_A(fake_A)
        loss_GAN_B2A = torch.sum((pred_fake - c) ** 2)
        loss_GAN_B2A = 0.5 * loss_GAN_B2A / self.batch_size

        # Cycle Consistency Loss
        recovered_A = self.netG_B2A(fake_B)
        loss_cycle_A = criterion_cycle(recovered_A, real_A)
        recovered_B = self.netG_A2B(fake_A)
        loss_cycle_B = criterion_cycle(recovered_B, real_B)
        loss_cycle = loss_cycle_A + loss_cycle_B

        # Identity Mapping Loss
        same_A = self.netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)
        same_B = self.netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)
        loss_identity = loss_identity_A + loss_identity_B

        # Total Loss
        loss_G = loss_GAN_A2B + loss_GAN_B2A + self.lambda_cycle * loss_cycle + self.lambda_identity * loss_identity

        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()
        
        # Logging.
        loss = {}
        loss['loss_GAN_A2B'] = loss_GAN_A2B.item()
        loss['loss_GAN_B2A'] = loss_GAN_B2A.item()
        loss['loss_cycle'] = loss_cycle.item()
        loss['loss_identity'] = loss_identity.item()
        loss['loss_G'] = loss_G.item()
        
        # ================================================================================ #
        #                             Train the discriminator                              #
        # ================================================================================ #

        ###### Discriminator A ######
        # Real loss
        pred_real = self.netD_A(real_A)
        loss_D_real = torch.sum((pred_real - b) ** 2)

        # Fake loss
        pred_fake = self.netD_A(fake_A.detach())
        loss_D_fake = torch.sum((pred_fake - a) ** 2)

        # Total loss
        loss_D_A = 0.5 * (loss_D_real + loss_D_fake) / self.batch_size

        self.optimizer_D_A.zero_grad()
        loss_D_A.backward()
        self.optimizer_D_A.step()

        ###### Discriminator B ######
        # Real loss
        pred_real = self.netD_B(real_B)
        loss_D_real = torch.sum((pred_real - b) ** 2)

        # Fake loss
        pred_fake = self.netD_B(fake_B.detach())
        loss_D_fake = torch.sum((pred_fake - a) ** 2)

        # Total loss
        loss_D_B = 0.5 * (loss_D_real + loss_D_fake) / self.batch_size

        self.optimizer_D_B.zero_grad()
        loss_D_B.backward()
        self.optimizer_D_B.step()
        
        # Logging.
        loss['loss_D_A'] = loss_D_A.item()
        loss['loss_D_B'] = loss_D_B.item()
        loss['loss_D'] = loss['loss_D_A'] + loss['loss_D_B']

        # Save
        if iters == max_iters:
            self.save_state(f'{epoch}_{iters}')
            img_name = str(epoch) + '_' + str(iters) + '.png'
            img_path = os.path.join(self.result_dir, img_name)
            self.save_sample(real_A, real_B, fake_A, fake_B, img_path)
            
        return loss
    
    def train(self, resume=True):
        self.netG_A2B.train()
        self.netG_B2A.train()
        self.netD_A.train()
        self.netD_B.train()
    
        max_iters = len(iter(self.dataloader))
        print(f'Max Iters: {max_iters}')
        
        for epoch in range(1, self.num_epochs + 1):
            if epoch < self.last_epoch:
                continue
            self.last_epoch = epoch

            self.epoch_loss_G = 0.0
            self.epoch_loss_D = 0.0
            
            for i, (data1, _) in enumerate(tqdm(self.dataloader)):
                data2 = []
                for d in data1:
                    data2.append(Util.toLineDrawing(d).numpy())
                data2 = torch.Tensor(data2).to(self.device)
                    
                data1 = data1.to(self.device)
                data2 = data2.to(self.device)
                
                loss = self.trainGAN(epoch, i + 1, max_iters, data1, data2)
                self.epoch_loss_G += loss['loss_G']
                self.epoch_loss_D += loss['loss_D']
                    
            print(f'{epoch} / {self.num_epochs}: Loss_G {self.epoch_loss_G}, Loss_D {self.epoch_loss_D}')
            
            if resume:
                self.save_resume()
            
        self.save_state('last')
              
    def save_sample(self, real_A, real_B, fake_A, fake_B, img_path):
        N = real_A.size(0)
        img = torch.cat((real_A.data, real_B.data, fake_A.data, fake_B.data), dim=0)
        save_image(img, img_path, nrow=N)
        #Util.showImages(fake_A, fake_B)
                
    def generate(self, num=1):
        self.netG_A2B.eval()
        self.netG_B2A.eval()

        dataloader = iter(self.dataloader)
        
        for i in range(num):
            data1, _ = next(dataloader)
            
            data2 = []
            for d in data1:
                data2.append(Util.toLineDrawing(d).numpy())
            data2 = torch.Tensor(data2).to(self.device)
                
            dataA = data1.to(self.device)
            dataB = data2.to(self.device)
            
            fakeB = self.netG_A2B(dataA).data
            fakeA = self.netG_B2A(dataB).data
            save_image(fakeA, os.path.join(self.result_dir, f'fakeA.{i + 1}.png'))
            save_image(fakeB, os.path.join(self.result_dir, f'fakeB.{i + 1}.png'))
            #Util.showImages(fakeA, fakeB)
