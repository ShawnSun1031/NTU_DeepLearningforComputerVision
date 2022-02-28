import torch
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import json 
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
from models import c3d, squeezenet, mobilenet, shufflenet, mobilenetv2, shufflenetv2, resnext, resnet
from scipy.ndimage import rotate
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p","--dataset_path",type=str,default="skull")
parser.add_argument("-d","--device",type=str,default="cuda: 0")
parser.add_argument("-z","--z_dim",type=int,default=48)
parser.add_argument("-m","--model_name",type=str)

args = parser.parse_args()
device = args.device
dsPath = args.dataset_path
channel = args.z_dim
modelName = args.model_name
img_size = 512
class SkullDataset(Dataset):
    def __init__(self, data_dir, split_indices ,classify = False, json_path=None, rotate=True):
        self.data_dir = data_dir
        self.split_indices = split_indices
        self.classify = classify
        self.json_path = json_path
        self.rotate = rotate
        with open(json_path) as f:
            self.json = json.load(f)
        self.len = len(self.split_indices)
    def __getitem__(self, index):
        root = os.path.join(self.data_dir,self.split_indices[index])
        imgs = []
        labels = []
        coords = []
        original_z = len(os.listdir(root))
        imgs = np.zeros([1,1,original_z,512,512]).astype("float")
        for i,imgName in enumerate(sorted(os.listdir(root))):
            name = imgName.replace(".npy","")
            labels.append(self.json["datainfo"][name]["label"])
            coords.append(self.json["datainfo"][name]["coords"])
            path = os.path.join(root,imgName)
            imgs[0,0,i,:,:] = np.load(path).astype("float")
        imgs = (imgs - imgs.min())/(imgs.max()-imgs.min())
        imgs = (imgs - 0.5) * 2.
        if self.rotate:
            angle = 40*(np.random.rand() - 0.5)
            imgs = rotate(imgs,angle,(3,4),reshape=False)
        imgs = torch.from_numpy(imgs)
        imgs = F.interpolate(imgs,(channel,img_size,img_size))[0]
        if self.classify:
            label = 0 if labels[-1] == 0 else 1
            return imgs,label
        else:
            return imgs,coords,labels
    def __len__(self):
        return self.len
BATCH_SIZE = 3
TIMES = 40
indices = os.listdir(os.path.join(dsPath,"train/"))
np.random.shuffle(indices)
val_indices = indices[:len(indices)//3]
train_indices = indices[len(indices)//3:]
trainSet = SkullDataset(os.path.join(dsPath,"train/"),train_indices,True,os.path.join(dsPath,"records_train.json"))
valSet = SkullDataset(os.path.join(dsPath,"train/"),val_indices,True,os.path.join(dsPath,"records_train.json"))
model = getattr(resnet,modelName)(
                num_classes=2,
                sample_size=img_size,
                sample_duration=channel)
model.conv1 = nn.Conv3d(1,64,kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
model.to(device)
trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, num_workers=4, drop_last=True, shuffle = True)
valLoader = DataLoader(valSet, batch_size=BATCH_SIZE, num_workers=4, drop_last=False, shuffle = False)
optimizer = optim.Adam(model.parameters(),lr=1e-4)
criterion = nn.CrossEntropyLoss()
EPOCH = 100
max_acc = 0.82
modelName = "3DCNN_" + modelName
if not os.path.isdir("model"):
    os.mkdir("model")
if not os.path.isdir("model/%s"%(modelName)):
    os.mkdir("model/%s"%(modelName))

END = ((len(trainSet) // BATCH_SIZE) // TIMES)*TIMES
for epoch in tqdm(range(EPOCH)):
    model.train()
    optimizer.zero_grad()
    ite = 0
    sample_loss = 0
    correct = 0
    total = 0
    for img,label in trainLoader:
        ite += 1
        img = img.float().to(device)
        label = label.long().to(device)
        pred = model(img)
        loss = criterion(pred,label)/float(TIMES)
        loss.backward()
        correct += np.sum(torch.max(pred,dim=1)[1].detach().cpu().numpy() == label.detach().cpu().numpy())
        total += pred.shape[0]
        sample_loss += loss.detach()*float(TIMES)
        if (ite+1) % TIMES == 0:
            # print(correct,total)
            optimizer.step()
            optimizer.zero_grad()
            sample_loss = 0
            if (ite+1) == END:
                break
    model.eval()
    with torch.no_grad():
        preds = []
        labels = []
        for (img,label) in valLoader:
            img = img.float().to(device)
            label = label.long().to(device)
            pred = model(img)
            preds.append(torch.max(pred,dim=1)[1].detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        acc = np.mean(preds == labels)
        precision = np.sum((preds == 1) * (labels == 1)) / np.sum(preds == 1)
        recall = np.sum((preds == 1) * (labels == 1)) / np.sum(labels == 1)
        f1 =  2*precision*recall / (precision+recall)
        print(float(correct)/float(total),acc,precision,recall,f1)
    if acc > max_acc:
        max_acc = acc
        torch.save(model.state_dict(),"model/%s/%.3f_%d.pth"%(modelName,acc,epoch))
torch.save(model.state_dict(),"model/%s/%.3f_%d.pth"%(modelName,acc,epoch))