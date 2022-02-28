# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 20:49:34 2021

@author: dicky1031
"""

import random
import torch.utils.data
import torchvision.utils as vutils

import torch.nn as nn


# Generator
class Generator(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(inputSize, hiddenSize*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hiddenSize*8),
            nn.ReLU(inplace=True),


            nn.ConvTranspose2d(hiddenSize*8, hiddenSize*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hiddenSize*4),
            nn.ReLU(inplace=True),


            nn.ConvTranspose2d(hiddenSize*4, hiddenSize*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hiddenSize*2),
            nn.ReLU(inplace=True),


            nn.ConvTranspose2d(hiddenSize*2, hiddenSize, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hiddenSize),
            nn.ReLU(inplace=True),


            nn.ConvTranspose2d(hiddenSize, outputSize, 4, 2, 1, bias=False),
            nn.Tanh())

    def forward(self, input):
        return self.main(input)

def saveimg(netG,save_file_dir):
    # Set random seed for reproducibility
    manualSeed = 123
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    netG.eval()
    noise = torch.randn(1000, 100, 1, 1, device=device)
    fake = netG(noise)
    count = 0
    for i in range(1000):
        vutils.save_image(fake[i], save_file_dir +'/'+str(count+1).zfill(4)+'.png', normalize=True)
        count+=1

if __name__ == '__main__':
    import sys
    save_file_dir = sys.argv[1]
    # CUDA
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # save 1000 img
    netG = torch.load('hw2_1_model_netG.pkl')
    saveimg(netG,save_file_dir)
    
    # FID 27.06  IS 2.21

