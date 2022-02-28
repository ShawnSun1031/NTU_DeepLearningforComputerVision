#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 15:56:31 2021

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
from torch.utils.data import ConcatDataset

import csv
import random
import numpy as np
import pandas as pd

from PIL import Image
filenameToPILImage = lambda x: Image.open(x)


#%% 
# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        class_id = -1
        wnid = []
        label = []
        for i in range(len(self.data_df)):
            nid =  self.data_df.loc[i, "label"]
            if nid not in wnid:
                wnid.append(nid)
                class_id += 1
            label.append(class_id)
        self.labels = label 

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.labels[index]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)

csv_path = 'hw4_data/mini/train.csv'
data_dir = 'hw4_data/mini/train'


batch_size = 64

train_dataset = MiniDataset(csv_path, data_dir)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle = True)

test_csv_path = 'hw4_data/mini/val.csv'
test_data_dir = 'hw4_data/mini/val'

valset = MiniDataset(test_csv_path, test_data_dir)
val_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle = True)

concat_data = ConcatDataset([train_dataset, valset])
concat_loader = DataLoader(dataset=concat_data, batch_size=batch_size, shuffle = True)

resnet = models.resnet50(pretrained=False).cuda()

learner = BYOL(
    resnet,
    image_size = 128,
    hidden_layer = 'avgpool'
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)


class Batch_Avg():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v
    
class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


timer = Timer()
trlog = {}
trlog['train_loss'] = []
max_epoch = 100
min_loss = 100

# for _ in range(100):
for epoch in tqdm(range(1,max_epoch + 1)):
    tl = Batch_Avg()
    addloss = 0
    for i, (images,label) in enumerate(concat_loader):
        
        images = images.cuda()
        loss = learner(images)
        addloss += loss.item()
        
        if i%200 == 0:
            print('epoch {}, train {}/{}, loss={:.4f} '
                      .format(epoch, i, len(concat_loader), loss.item()))
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() # update moving average of target encoder  
        tl.add(loss.item())
        
    
        
    tl = tl.item()
    addloss = addloss / len(concat_loader)
    if addloss < min_loss:
        min_loss = addloss
        torch.save(resnet.state_dict(), f'{addloss}improved-net.pt')
    
    print('epoch {}, train, loss={:.4f}'.format(epoch, tl)) 
    trlog['train_loss'].append(tl)
    
    print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / max_epoch)))
    time.sleep(0.01)
    
# save your improved network
torch.save(resnet.state_dict(), f'{loss}improved-net.pt')
torch.save(trlog, 'trlog')
# a = torch.load('improved-net.pt')



