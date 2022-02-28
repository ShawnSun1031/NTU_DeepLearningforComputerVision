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
from models import c3d, squeezenet, mobilenet, shufflenet, mobilenetv2, shufflenetv2, resnext, resnet
import torch.nn.functional as F
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p","--dataset_path",type=str,default="skull")
parser.add_argument("-d","--device",type=str,default="cuda: 0")
parser.add_argument("-m","--num_models",type=int,default=3)
args = parser.parse_args()
device = args.device
dsPath = args.dataset_path
num_models = args.num_models

class SkullDataset(Dataset):
    def __init__(self, data_dir,z_dim=48):
        self.data_dir = data_dir
        self.split_indices = sorted(os.listdir(data_dir))
        self.len = len(self.split_indices)
        self.z_dim= z_dim
    def __getitem__(self, index):
        root = os.path.join(self.data_dir,self.split_indices[index])
        imgs = []
        original_z = len(os.listdir(root))
        imgs = np.zeros([1,1,original_z,512,512]).astype("float")
        for i,imgName in enumerate(sorted(os.listdir(root))):
            name = imgName.replace(".npy","")
            path = os.path.join(root,imgName)
            imgs[0,0,i,:,:] = np.load(path).astype("float")
        imgs = (imgs - imgs.min())/(imgs.max()-imgs.min())
        imgs = (imgs - 0.5) * 2.
        imgs = torch.from_numpy(imgs)
        imgs = F.interpolate(imgs,(self.z_dim,512,512))[0]
        return imgs
    def __len__(self):
        return self.len
def pred(kind,modelName,modelPath,z_dim=48,dsPath="skull"):
    valSet = SkullDataset(os.path.join(dsPath,"%s/"%(kind)),z_dim)
    BATCH_SIZE = 1
    valLoader = DataLoader(valSet, batch_size=BATCH_SIZE, num_workers=4, drop_last=False, shuffle = False)
    model = getattr(resnet,modelName)(
                    num_classes=2,
                    sample_size=512,
                    sample_duration=z_dim)
    model.conv1 = nn.Conv3d(1,64,kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
    model.to(device)
    model.load_state_dict(torch.load(modelPath,map_location=device))
    model.eval()
    with torch.no_grad():
        preds = []
        for img in tqdm(valLoader):
            img = img.float().to(device)
            pred = model(img)
            preds.append(torch.max(pred,dim=1)[1].detach().cpu().numpy())    
        preds = np.concatenate(preds)
    f = open("pred/%s_%s.csv"%(modelName,kind),"w")
    f.write("id,label,coords\n")
    for i,p in enumerate(preds):
        for fn in sorted(os.listdir(os.path.join(valSet.data_dir,valSet.split_indices[i]))):
            fn = fn.replace(".npy","")
            if p == 1:
                f.write(fn+",1,0 0")
            else:
                f.write(fn+",0,")
            f.write("\n")
    f.close()
    del model
if num_models == 3:
    models = {
         "resnet18":("model/3DCNN_resnet18/0.859_47.pth",48),
         "resnet34":("model/3DCNN_resnet34/0.862_26.pth",48),
         "resnet50":("model/3DCNN_resnet50/0.871_59.pth",48),}
else:
    models = {"resnet101":("model/3DCNN_resnet101/0.849_60.pth",36),
         "resnet18":("model/3DCNN_resnet18/0.859_47.pth",48),
         "resnet34":("model/3DCNN_resnet34/0.862_26.pth",48),
         "resnet50":("model/3DCNN_resnet50/0.871_59.pth",48),}

         
# for kind in ["train","test"]:
for kind in ["test"]:
    for key in models.keys():
        pred(kind,key,*models[key],dsPath=dsPath)
num_of_models = len(models.keys())
if num_of_models % 2 == 1:
    N = num_of_models // 2 + 1
else:
    N = num_of_models //2
# N : voting threshold
#for kind in ["train","test"]:
for kind in ["test"]:
    df = []
    for model in models.keys():
        if len(df) == 0:
            df = pd.read_csv("pred/%s_%s.csv"%(model,kind))
        else:
            df["label"] =  df["label"] + pd.read_csv("pred/%s_%s.csv"%(model,kind))["label"]
    df["label"] = (df["label"] >= N).astype("int")
    df.to_csv("pred/clf_%s_%d.csv"%(kind,num_models),index=0)
