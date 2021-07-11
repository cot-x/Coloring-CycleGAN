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
from solver import *


def main(args):
    hyper_params = {}
    hyper_params['Image Dir'] = args.img_dir
    hyper_params['Image Size'] = args.img_size
    hyper_params['Result Dir'] = args.result_dir
    hyper_params['Wieght Dir'] = args.weight_dir
    hyper_params['Learning Rate'] = args.lr
    hyper_params['Epochs'] = args.num_epoch
    hyper_params['Batch Size'] = args.batch_size
    hyper_params['lambda_cycle'] = args.lambda_cycle
    hyper_params['lambda_identity'] = args.lambda_identity
    
    solver = Solver(args.cpu, args.lr, args.num_epoch, args.batch_size, args.img_dir, args.img_size,
                    args.lambda_cycle, args.lambda_identity, args.result_dir, args.weight_dir)
    solver.load_state()
    
    for key in hyper_params.keys():
        print(f'{key}: {hyper_params[key]}') 
    
    if not args.noresume:
        solver = solver.load_resume()
    
    if args.generate != 0:
        solver.generate(args.generate)
        exit()
    
    #Util.showImages(solver.dataloader)
    solver.train(not args.noresume)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='../datasets/')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--weight_dir', type=str, default='weights')
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lambda_cycle', type=float, default=10)
    parser.add_argument('--lambda_identity', type=float, default=5)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--generate', type=int, default=0)
    parser.add_argument('--noresume', action='store_true')

    args, unknown = parser.parse_known_args()
    
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    if not os.path.exists(args.weight_dir):
        os.mkdir(args.weight_dir)
        
    if args.generate:
        args.batch_size = 1
    
    main(args)
