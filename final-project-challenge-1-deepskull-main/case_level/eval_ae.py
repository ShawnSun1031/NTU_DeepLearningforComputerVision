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
from monai.networks.nets import  AutoEncoder
import subprocess
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p","--model_path",type=str,default="skull")
parser.add_argument("-o","--out_path",type=str,default="skull")

args = parser.parse_args()
modelPath = args.model_path
outPath = args.out_path
# https://towardsdatascience.com/hands-on-anomaly-detection-with-variational-autoencoders-d4044672acd5
class SkullDataset(Dataset):
    def __init__(self, data_dir, split_indices ,json_path=None, is_train=True, is_pos=False):
        self.data_dir = data_dir
        self.split_indices = split_indices
        self.json_path = json_path
        self.is_train = is_train
        with open(json_path) as f:
            self.json = json.load(f)
        self.len = len(self.split_indices)
        self.is_pos = is_pos
    def __getitem__(self, index):
        root = os.path.join(self.data_dir,self.split_indices[index])
        labels = []
        z_dim = len(os.listdir(root))

        imgs = np.empty([z_dim,512,512],dtype=np.float64)
        start = 0
        end = z_dim

        img_paths = sorted(os.listdir(root))
        for i in range(start,end):
            name = img_paths[i].replace(".npy","")
            path = os.path.join(root,img_paths[i])
            imgs[i-start,:,:] = np.load(path)
            
        imgs = (imgs + 1024.) / 4095.
#         imgs /= 10.
#         imgs = np.where(imgs < 0, 0, imgs) 
#         imgs = np.where(imgs > 255, 255,imgs) 
#         imgs /= 255.
#         imgs = (imgs - imgs.min())/(imgs.max()-imgs.min())
#         imgs = (imgs - 0.5) * 2.
        out_imgs = np.zeros([z_dim-15,1,16,512,512]).astype("float")
        for i in range(z_dim-15):
            out_imgs[i,0,:,:,:] = imgs[i:i+16,:,:]
        out_imgs = torch.from_numpy(out_imgs)
        
        return out_imgs.float(),img_paths
    def __len__(self):
        return self.len
validset = SkullDataset("skull/test/", sorted(os.listdir("skull/test/")), "skull/records_train.json",False,False)
valLoader = DataLoader(validset, batch_size=1, num_workers=4, drop_last=False, shuffle = False)
# valIter = iter(valLoader)
# imgs = next(valIter)
model = AutoEncoder(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(32,32,64,64,128,256),
            strides=((1,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),(1,2,2)),
            kernel_size=(3,3,3),
            up_kernel_size=(3,3,3),
        ) 

model.cuda()
L1Loss = torch.nn.L1Loss(reduction='mean')
loss_function = L1Loss
model.load_state_dict(torch.load(modelPath))
sig = nn.Sigmoid()
# for t in [0.6]:
for t in [0.6]:#np.linspace(0,1,11)[3:]:
    model.eval()
    out = "id,label,coords\n"
    with torch.no_grad():
        for imgs,img_paths in tqdm(valLoader):
            outSet = set()
            imgs = imgs.cuda()
            pred = sig(model(imgs[0]))
            recon_loss = torch.abs(pred - imgs[0])
            flatten = recon_loss.view(recon_loss.shape[0],-1)
            value,indexs = torch.sort(flatten,dim=1,descending=True)
            value = value
            indexs = indexs
            zs = torch.clone(indexs)//(512*512)
            indexs -= zs*(512*512)
            ys = torch.clone(indexs)//512
            indexs -= ys*512
            xs = torch.clone(indexs)
            zs += torch.arange(imgs.shape[1]).reshape(-1,1).cuda()
            mask = (value > t) * ((imgs[0] < pred).view(imgs.shape[1],-1))
            zs = zs[mask].detach().cpu().numpy()
            ys = ys[mask].detach().cpu().numpy()
            xs = xs[mask].detach().cpu().numpy()
            for z,y,x in zip(zs,ys,xs):
                outSet.add((z,x,y))
            outList = list(outSet)
            outList.sort(key = lambda x:x[0])
            j = 0
            for i in range(len(img_paths)):
                app = [str(img_paths[i][0]).replace(".npy","")]
                if j >= len(outList) or outList[j][0] != i:
                    app.append("-1,")
                else:
                    app.append("1")
                    coordstxt = ""
                    while j < len(outList) and outList[j][0] == i:
                        if coordstxt != "":
                            coordstxt += " "
                        coordstxt += "%d %d"%(outList[j][1],outList[j][2])
                        j += 1
                    app.append(coordstxt)
                out += ",".join(ele for ele in app)
                out += "\n"
    with open(outPath,"w") as f:
        f.write(out)    
