#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 18:50:34 2021

@author: md703
"""
import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
import os
from PIL import Image
import torch
from torchvision import transforms
import pandas as pd
import sys

"""Load Data"""
class hw3_data(Dataset):
    def __init__(self,root,transform=None):
        self.root = root
        self.transform = transform
        self.images = None
        self.filenames = []
        
        #read filenames
        file_list = [file for file in os.listdir(self.root)]
        for name in file_list:
            filename = os.path.join(self.root, name)
            self.filenames.append((filename,name))
        self.len = len(self.filenames)

    
    def __getitem__(self,index):
        image_fn,name = self.filenames[index]
        image = Image.open(image_fn).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image).float()
    
        return image,name
    
    def __len__(self):
        return self.len 
    
model = torch.load('hw3_1_model.pth')

test_file_dir = sys.argv[1]
save_file_dir = sys.argv[2]
trans = transforms.Compose([
                        transforms.Resize([384,384]),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
                        ])
test_dataset = hw3_data(test_file_dir, transform=trans)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

model.eval()
## save csv and calculate acc
correct = 0
result = []
img_name = []
for i,(img,name) in enumerate(test_dataloader):
    img = img.cuda()
    output = model(img)
    pred = torch.max(output,1)[1]
    pred_label = pred.detach().cpu().numpy()
    result.append(pred_label)
    img_name.append(name)
result = np.concatenate(result)
img_name = np.concatenate(img_name)
df = pd.DataFrame({'filename': img_name, 'label': result})
df.to_csv(save_file_dir,index=False)    
