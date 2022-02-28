# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 21:03:21 2021

@author: dicky1031
"""

import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
import pandas as pd
import torch.nn as nn


class hw2_bonus(Dataset):
    def __init__(self,root,transform=None):
        self.root = root
        self.transform = transform
        self.filenames = []
        
        #read filenames
        file_list = [file for file in os.listdir(root) if file.endswith('.png')]
        file_list.sort()

        for i, file in enumerate(file_list):
            filename = os.path.join(root, file)
        # for i in range(df["image_name"].size):
            # filename = self.root + "/" + str(df["image_name"][i])
            # label = int(df["label"][i])
            self.filenames.append((filename , file))
            # print(str(df["image_name"][i]) , label)
            # print(filename , label)
        self.len = len(self.filenames)

    
    def __getitem__(self,index):
        image_fn,_ = self.filenames[index]
        image = Image.open(image_fn)
        # image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image).float()
        
        return image
    
    def __len__(self):
        return self.len 

 
class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
    def forward(self, x):
        x = self.conv(x).squeeze()
        # print(x.size())
        return x

class LabelPredictor(nn.Module):

    def __init__(self):
        super(LabelPredictor, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c

if __name__ == '__main__':
    
    import sys
    test_img_dir = sys.argv[1]
    target_name = sys.argv[2]
    save_img_dir = sys.argv[3]
    
    target_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    test_dataset = hw2_bonus(test_img_dir, transform=target_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
       
    
    
    label_predictor = torch.load(target_name+'_bonus_predictor_model.pth')
    feature_extractor = torch.load(target_name+'_bonus_extractor_model.pth')
    result = []
    label_predictor.eval()
    feature_extractor.eval()
    
    ## save csv and calculate acc
    correct = 0
    for i, test_data in enumerate(test_dataloader):
        test_data = test_data.cuda()
     
        class_logits = label_predictor(feature_extractor(test_data))
        pred = class_logits.max(1,keepdim=True)[1]
        
    
        x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
        result.append(x)
    # print('acc: ',correct/len(test_dataset)*100)
    
    result = np.concatenate(result)
    
    # Generate your submission
    img_name = []
    for i in range(0,len(result)):
        img_name.append(str(i).zfill(5)+'.png')
        
    df = pd.DataFrame({'image_name': img_name, 'label': result})
    df.to_csv(save_img_dir,index=False)