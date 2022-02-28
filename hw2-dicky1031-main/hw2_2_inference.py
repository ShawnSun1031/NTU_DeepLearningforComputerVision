# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 20:55:00 2021

@author: dicky1031
"""


import random
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import torch.nn as nn


G_in = 100
num_class = 10
# Generator
class Generator(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_class, G_in)  
        self.init_size = 8  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(G_in, 128 * self.init_size ** 2))
        
        self.main = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
          
            nn.ConvTranspose2d(128, hiddenSize*4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hiddenSize*4),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.5),

            nn.ConvTranspose2d(hiddenSize*4, hiddenSize*2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hiddenSize*2),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.5),

            nn.ConvTranspose2d(hiddenSize*2, hiddenSize, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hiddenSize),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.5),

            nn.ConvTranspose2d(hiddenSize, outputSize, 4, 1, 2, bias=False),
            nn.Tanh())

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        # print('gen: ',gen_input.size()) #batch*Gin
        out = self.l1(gen_input) 
        # print('out1',out.size()) #batch*128*4**2
        out = out.view(out.shape[0], 128, self.init_size, self.init_size) 
        # print('out2',out.size()) #batch*128*4*4
        img = self.main(out) 
        # print('mani',img.size()) #batch*3*28*28
        return img


def saveimg(netG,save_file_dir):
    # Set random seed for reproducibility
    manualSeed = 223
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    netG.eval()
    noise = torch.tensor(np.random.normal(0, 1, (1000, 100))).to(device).float()
    list_label = []
    for i in range(10):
        for j in range(100):
            list_label.append(i) 
    label = torch.tensor(list_label).to(device)
    fake = netG(noise,label)
    for i in range(10):
        for j in range(100):
            vutils.save_image(fake[100*i+j], save_file_dir +'/'+str(i)+'_'+str(j+1).zfill(3)+'.png', normalize=True)

if __name__ == '__main__':
    import sys
    save_file_dir = sys.argv[1]
    # CUDA
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    netG = torch.load('hw2_2_model_netG.pkl')
    saveimg(netG,save_file_dir)


    
 
