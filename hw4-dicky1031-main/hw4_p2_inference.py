#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 00:06:02 2021

@author: md703
"""

import os
import sys
import argparse
import time
from tqdm import tqdm

from byol_pytorch import BYOL
from torchvision import models

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F

import csv
import random
import numpy as np
import pandas as pd

from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

#%% load data
class Hometest(Dataset):
    def __init__(self, csv_path, data_dir,label_dict):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")
        
        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.Resize([128,128]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        
        self.label_dict = label_dict

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, path

    def __len__(self):
        return len(self.data_df)



#%% load model
class Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=False)
        self.resnet = resnet
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,65)
            )
    
    def forward(self,x):
        x = self.resnet(x)
        x = self.fc(x)
        
        return x
    
#%% label num2str
def get_key(val):
    for key, value in label_dict.items():
        if val == value:
            return key
    return "There is no such Key"    
#%%
if __name__ == '__main__':
    
    test_csv_file = sys.argv[1]
    img_file_dir = sys.argv[2]
    out_csv_file = sys.argv[3]
    
    batch_size = 8
    
    test_csv_path = test_csv_file
    test_data_dir = img_file_dir
    
    label_dict = np.load('hw4_p2_label_dict.npy', allow_pickle='TRUE').item()
    valset = Hometest(test_csv_path, test_data_dir,label_dict)
    val_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle = False)    
    
    
    model = Resnet().cuda()
    model.load_state_dict(torch.load("hw4_p2_model.pth"))
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    model.eval()
    numpred = []
    img_name = []
    for i, (images,name) in enumerate(val_loader):
        images = images.cuda()
        img_name.append(name)
        
        pred = model(images)
        pred_label = torch.argmax(pred, dim=1)
        numpred.append(pred_label.cpu().numpy())
        
    numpred = np.concatenate(numpred)
    img_name = np.concatenate(img_name)
    header = []
    strlabel = []
    strpred = []
    for i in range(len(valset)):
        header.append(i)
        strpred.append(get_key(numpred[i]))
        
    df = pd.DataFrame({'id':header,'filename':img_name,'label':strpred})
    df.to_csv(out_csv_file,index=False)
      

