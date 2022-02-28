#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 17:12:39 2021

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
class Home(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")
        
        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.Resize([128,128]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop([100,100])
            ])
        
        class_id = -1
        label_dict = {}
        label = []
        for i in range(len(self.data_df)):
            nid =  self.data_df.loc[i, "label"]
            if nid not in label_dict:
                class_id += 1
                label_dict[nid] = class_id
            label.append(label_dict[nid])
        self.labels = label 
        self.label_dict = label_dict

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.labels[index]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)
    
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
        

        label = []
        for i in range(len(self.data_df)):
            nid =  self.data_df.loc[i, "label"]
            label.append(label_dict[nid])
        self.labels = label 
        self.label_dict = label_dict

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.labels[index]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)
    

csv_path = 'hw4_data/office/train.csv'
data_dir = 'hw4_data/office/train'

batch_size = 64

train_dataset = Home(csv_path, data_dir)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle = True)


test_csv_path = 'hw4_data/office/val.csv'
test_data_dir = 'hw4_data/office/val'

label_dict = train_dataset.label_dict
valset = Hometest(test_csv_path, test_data_dir,label_dict)
val_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle = False)

#%% fullmodel
pretrained_weight = torch.load('backbone_improved-net.pt')
resnet = models.resnet50(pretrained=False).cuda()
resnet.load_state_dict(pretrained_weight)

class Resnet(nn.Module):
    def __init__(self):
        super().__init__()
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

# class classifier(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = nn.Sequential(
#                     nn.ReLU(),
#                     nn.Linear(1000, 512),
#                     nn.ReLU(),
#                     nn.Linear(512,256),
#                     nn.ReLU(),
#                     nn.Linear(256,65)
#                     )
            
        
#     def forward(self,x):
#         x = self.linear(x)
        
#         return x
        
fullmodel = Resnet().cuda() 
 
# classi = classifier().cuda() 
opt = torch.optim.Adam(fullmodel.parameters(), lr=3e-4)

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

def save_model(name):
    torch.save(fullmodel.state_dict(), (name + '.pth'))

trlog = {}
trlog['train_loss'] = []
trlog['val_loss'] = []
trlog['train_acc'] = []
trlog['val_acc'] = [] 

timer = Timer()
max_epoch = 50
max_acc = 0
save_epoch = 10

criterion = nn.CrossEntropyLoss()
for epoch in tqdm(range(1,max_epoch + 1)):
    fullmodel.train()
    # resnet.train()
    # classi.train()
    tl = Batch_Avg()
    ta = Batch_Avg()
    
    for i, (images,label) in enumerate(train_loader):
        
        images,label = images.cuda(),label.cuda()
        
        pred = fullmodel(images)
        # backbone = resnet(images)
        # pred = classi(backbone)
        
        loss = criterion(pred, label)
        
        pred_label = torch.argmax(pred, dim=1)
        acc = (pred_label == label).type(torch.cuda.FloatTensor).mean().item()
        
        if i%31 == 0:
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                      .format(epoch, i, len(train_loader), loss.item(), acc))
            
        tl.add(loss.item())
        ta.add(acc)   
         
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        loss = None; pred = None
    
    tl = tl.item()
    ta = ta.item()
    print('epoch {}, train, loss={:.4f} acc={:.4f}'.format(epoch, tl, ta)) 
    
    fullmodel.eval()
    # resnet.eval()
    # classi.eval()
    
    vl = Batch_Avg()
    va = Batch_Avg()
    correct = 0
    for i, (images,label) in enumerate(val_loader):
        
        images,label = images.cuda(),label.cuda()
        
        pred = fullmodel(images)
        # backbone = resnet(images)
        # pred = classi(backbone)
        loss = criterion(pred, label)
        
        pred_label = torch.argmax(pred, dim=1)
        correct += (pred_label == label).type(torch.cuda.FloatTensor).sum().item()

            
        vl.add(loss.item())
        va.add(correct)
        
        loss = None; pred = None

    vl = vl.item()
    va = va.item()
    vacc = correct/len(valset)
    print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, vacc))    
    
    if vacc > max_acc:
            max_acc = vacc
            save_model('{}-epoch-max-acc={:.4f}'.format(epoch,vacc))


    trlog['train_loss'].append(tl)
    trlog['train_acc'].append(ta)
    trlog['val_loss'].append(vl)
    trlog['val_acc'].append(vacc)

    torch.save(trlog, 'trlog')

    if epoch % save_epoch == 0:
        save_model('epoch-{}-acc={:.4f}'.format(epoch,vacc))

    print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / max_epoch)))
    time.sleep(0.01)
    
    
    #p2_1 acc = 19.21%
    #p2_2 acc = 31.53%
    #p2_3 acc = 37.93%
    #p2_4 acc = 23.89%








