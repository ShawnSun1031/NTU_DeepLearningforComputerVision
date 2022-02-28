#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 16:16:44 2021

@author: md703
"""
import os
import sys
import argparse
import time
from tqdm import tqdm

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
 
#%% 
# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
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

class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1).tolist()
            
            yield batch

csv_path = 'hw4_data/mini/train.csv'
data_dir = 'hw4_data/mini/train'

train_way = 30
test_way = 5
query = 15
shot = 1
batch_size = 200

train_dataset = MiniDataset(csv_path, data_dir)
train_sampler = CategoriesSampler(train_dataset.labels, batch_size,train_way, shot + query)
train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler)

test_csv_path = 'hw4_data/mini/val.csv'
test_data_dir = 'hw4_data/mini/val'

valset = MiniDataset(test_csv_path, test_data_dir)
val_sampler = CategoriesSampler(valset.labels, 600,test_way, shot + query)
val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler)



#%% model
def conv_block(in_channel , out_channel):
    bn = nn.BatchNorm2d(out_channel)
    return nn.Sequential(
        nn.Conv2d(in_channel , out_channel ,3 ,padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
        )

class Convnet(nn.Module):
    def __init__(self , in_channel = 3 , hid_channel = 64 , out_channel = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channel , hid_channel),
            conv_block(hid_channel , hid_channel),
            conv_block(hid_channel , hid_channel),
            conv_block(hid_channel , hid_channel),
            )
    def forward(self,x):
        x = self.encoder(x)
       
        return x.view(x.size(0),-1)


model = Convnet().cuda()
 
# model.load_state_dict(torch.load('30way-ecludian-max-acc=0.4553.pth'))
    
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

def save_model(name):
    # torch.save(model.state_dict(), (name + '.pth'))
    torch.save(model.state_dict(), (name + '.pth'))
    
def save_dmodel(name):
    torch.save(d_model.state_dict(), (name + '.pth'))

# for i, batch in enumerate(train_loader):

#%%
def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def cos_metric(a,b):
    n_x = a.shape[0]
    n_y = b.shape[0]
    normalised_x = a / (a.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)
    normalised_y = b / (b.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)

    expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
    expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

    cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
    return cosine_similarities

class distance(nn.Module):
    def __init__(self,out_channel=5,in_channel=5,hid_channel=256):
        super().__init__()
        self.out_channel = out_channel
        self.linear1 = nn.Sequential(
            nn.Linear(in_channel, hid_channel*4),
            nn.ReLU(),
            nn.Linear(hid_channel*4, hid_channel*3),
            nn.ReLU(),
            nn.Linear(hid_channel*3, hid_channel*2),
            nn.ReLU(),
            nn.Linear(hid_channel*2, hid_channel),
            nn.ReLU(),
            nn.Linear(hid_channel, 128),
            nn.ReLU(),
            nn.Linear(128, out_channel),
            )
    def forward(self,a,b):
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n,m,-1)
        b = b.unsqueeze(0).expand(n,m,-1)
        x = ((a - b)**2).sum(dim=2)
        x = self.linear1(x)
  
        return x

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

d_model = distance().cuda()
d_optimizer = torch.optim.Adam(d_model.parameters(), lr=0.001)
d_lr_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=20, gamma=0.5)
trlog = {}
trlog['train_loss'] = []
trlog['val_loss'] = []
trlog['train_acc'] = []
trlog['val_acc'] = [] 

timer = Timer()

max_epoch = 200
max_acc = 0
save_epoch = 20
for epoch in tqdm(range(1,max_epoch + 1)):
    lr_scheduler.step()
    d_lr_scheduler.step()
    model.train()
    d_model.train()
    tl = Batch_Avg()
    ta = Batch_Avg()

    for i, batch in enumerate(train_loader, 1):
        data, _ = [_.cuda() for _ in batch]
        p = shot * train_way
        data_shot, data_query = data[:p], data[p:]

        proto = model(data_shot)
        proto = proto.reshape(shot, train_way, -1).mean(dim=0)

        label = torch.arange(train_way).repeat(query)
        label = label.type(torch.cuda.LongTensor)
        
        logits = euclidean_metric(model(data_query), proto)
        # logits = d_model(model(data_query), proto)
        loss = F.cross_entropy(logits, label)
        
        pred = torch.argmax(logits, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
        
        if i%50 == 0:
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                      .format(epoch, i, len(train_loader), loss.item(), acc))
        
        tl.add(loss.item())
        ta.add(acc)
        
        optimizer.zero_grad()
        d_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        d_optimizer.step()
       
        proto = None; logits = None; loss = None
    
    tl = tl.item()
    ta = ta.item()
    
    model.eval()
    d_model.eval()

    vl = Batch_Avg()
    va = Batch_Avg()
    
    for i, batch in enumerate(val_loader, 1):
        data, _ = [_.cuda() for _ in batch]
        p = shot * test_way
        data_shot, data_query = data[:p], data[p:]

        proto = model(data_shot)
        proto = proto.reshape(shot, test_way, -1).mean(dim=0)

        label = torch.arange(test_way).repeat(query)
        label = label.type(torch.cuda.LongTensor)

        # logits = d_model(model(data_query), proto)
        logits = euclidean_metric(model(data_query), proto)
        loss = F.cross_entropy(logits, label)
        
        pred = torch.argmax(logits, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
        
        vl.add(loss.item())
        va.add(acc)
        
        proto = None; logits = None; loss = None
  
    vl = vl.item()
    va = va.item()
    print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))    
    
    if va > max_acc:
            max_acc = va
            save_model('{}-epoch-max-acc={:.4f}'.format(epoch,va))
            # save_dmodel('d-{}-epoch-max-acc={:.4f}'.format(epoch,va))


    trlog['train_loss'].append(tl)
    trlog['train_acc'].append(ta)
    trlog['val_loss'].append(vl)
    trlog['val_acc'].append(va)

    torch.save(trlog, 'trlog')

    # save_model('epoch-last-acc={:.4f}'.format(va))

    if epoch % save_epoch == 0:
        save_model('epoch-{}-acc={:.4f}'.format(epoch,va))

    print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / max_epoch)))
    time.sleep(0.01)
    # a = torch.load('trlog')
    