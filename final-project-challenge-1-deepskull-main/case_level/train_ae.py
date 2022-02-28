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
import random
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p","--dataset_path",type=str,default="skull")
args = parser.parse_args()
dsPath = args.dataset_path

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
        self.coords = np.load(os.path.join("supplement/coords.npy"),allow_pickle=True).item()
        self.is_pos = is_pos
    def __getitem__(self, index):
        root = os.path.join(self.data_dir,self.split_indices[index])
        labels = []
        z_dim = len(os.listdir(root))
        imgs = np.empty([1,16,512,512],dtype=np.float64)
        if self.is_train:
            if self.is_pos:
                sampled_z = random.sample(self.coords[self.split_indices[index]],1)[0][0]
                if sampled_z + 16 > z_dim:
                    end = z_dim
                    start = z_dim - 16
                elif sampled_z < 16:
                    start = 0
                    end = 16
                else:
                    start = np.random.randint(max(sampled_z-15,0),sampled_z)
                    end = start + 16
                coords =  np.array(sorted(list(self.coords[self.split_indices[index]]),key=lambda l:l[0], reverse=False),dtype=np.int)
                coords = coords[(coords[:,0] >= start) * (coords[:,0] < end)]
                coords[:,0] -= start
                out_coords = np.zeros([200,3])
                out_coords[:coords.shape[0],:] = coords
                out_coords[min(coords.shape[0],199),0] = -1
            else:
                end = np.random.randint(16,z_dim)
                start = end - 16
        img_paths = sorted(os.listdir(root))
        for i in range(start,end):
            name = img_paths[i].replace(".npy","")
            labels.append(self.json["datainfo"][name]["label"])
            path = os.path.join(root,img_paths[i])
            imgs[0,i-start,:,:] = np.load(path)
        imgs = (imgs + 1024.) / 4095.
        # imgs /= 10.
        # imgs = np.where(imgs < 0, 0, imgs) 
        # imgs = np.where(imgs > 255, 255,imgs) 
        # imgs /= 255.
        # imgs = (imgs - imgs.min())/(imgs.max()-imgs.min())
        # imgs = (imgs - 0.5) * 2.
        imgs = torch.from_numpy(imgs)
        if self.is_pos:
            out_coords = torch.from_numpy(out_coords)
            return imgs.float(),out_coords.long()
        return imgs.float()
    def __len__(self):
        return self.len
BATCH_SIZE = 16
device = "cuda: 0"
LAMBDA = 0.01
modelName = "compAE_%.3f_norm"%(LAMBDA)
if not os.path.isdir("model/%s"%(modelName)):
    os.mkdir("model/%s"%(modelName))
train_path = os.path.join(dsPath,"train")
record_path = os.path.join(dsPath,"records_train.json")
train_pos_indices,train_neg_indices,val_pos_indices,val_neg_indices = np.load("supplement/split.npy",allow_pickle=True)
negSet = SkullDataset(train_path,train_neg_indices , record_path,True,False)
posSet = SkullDataset(train_path,train_pos_indices , record_path,True,True)
valNegSet = SkullDataset(train_path,val_neg_indices , record_path,True,False)
valPosSet = SkullDataset(train_path,val_pos_indices , record_path,True,True)

posLoader = DataLoader(posSet, batch_size=BATCH_SIZE, num_workers=4, drop_last=True, shuffle = True)
negLoader = DataLoader(negSet, batch_size=BATCH_SIZE//4, num_workers=4, drop_last=True, shuffle = True)
valPosLoader = DataLoader(valPosSet, batch_size=BATCH_SIZE, num_workers=4, drop_last=True, shuffle = True)
valNegLoader = DataLoader(valNegSet, batch_size=BATCH_SIZE//4, num_workers=4, drop_last=True, shuffle = True)

posIter = iter(posLoader)
negIter = iter(negLoader)
valPosIter = iter(valPosLoader)
valNegIter = iter(valNegLoader)

model = AutoEncoder(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(32,32,64,64,128,256),
            strides=((1,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),(1,2,2)),
            kernel_size=(3,3,3),
            up_kernel_size=(3,3,3),
        ) 
model.to(device)
BATCH_SIZE = 8
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
L1Loss = torch.nn.L1Loss(reduction='mean')
loss_function = L1Loss
print(model.encode(torch.rand(1,1,16,512,512).to(device)).shape)
print(model(torch.rand(1,1,16,512,512).to(device)).shape)
EPOCH = 10000
max_loss = None
sig = nn.Sigmoid()
val_step = 200
for epoch in range(EPOCH):
    model.train()
    training_loss = 0.
    optimizer.zero_grad()
    # neg sampling
    try:
        data = next(negIter)
    except:
        negIter = iter(negLoader)
        data = next(negIter)
    data = data.to(device)
    recon_batch = sig(model(data))
    recon_loss = loss_function(recon_batch, data)
    recon_loss.backward()
    training_loss += recon_loss.detach()
    if (epoch + 1) % val_step == 0:
        plt.imsave("model/%s/reconstruct.png"%(modelName),recon_batch[0,0,4].detach().cpu().numpy())
        plt.imsave("model/%s/truth.png"%(modelName),data[0,0,4].detach().cpu().numpy())
#         plt.imshow(recon_batch[0,0,2].detach().cpu().numpy())
#         plt.show()
#         plt.imshow(data[0,0,2].detach().cpu().numpy())
#         plt.show()

    # pos sampling
    try:
        data,coords = next(posIter)
    except:
        posIter = iter(posLoader)
        data,coords = next(posIter)
    data = data.to(device)
    recon_batch = sig(model(data))
    comp_pos_loss = 0.
    comp_neg_loss = 0.
    count = 0.
    for i in range(data.shape[0]):
        pos_coords = set()
        instance_count = 0.
        for (z,y,x) in coords[i]:
            if z == -1:
                break
            pos_coords.add((z,y,x))
            comp_pos_loss -= loss_function(data[i,0,z,y,x],recon_batch[i,0,z,y,x])
            instance_count += 1
        count += instance_count
        j = 0
        while j < instance_count:
            z = np.random.randint(0,16)
            y = np.random.randint(0,512)
            x = np.random.randint(0,512)
            if (z,y,x) in pos_coords:
                continue
            else:
                j += 1
                comp_neg_loss += loss_function(data[i,0,z,y,x],recon_batch[i,0,z,y,x])
    comp_pos_loss /= float(count)
    comp_neg_loss /= float(count)
    comp_loss = LAMBDA *(comp_pos_loss+comp_neg_loss)
    comp_loss.backward()
    training_loss += comp_loss.detach()
    optimizer.step()
    with torch.no_grad():
        if (epoch + 1) % val_step == 0:
            print("%d training loss = %.3f %.3f %.3f %.3f"%(epoch,training_loss,recon_loss,comp_pos_loss,comp_neg_loss))
            model.eval()
            try:
                data = next(valNegIter)
            except:
                valNegIter = iter(valNegLoader)
                data = next(valNegIter)
            data = data.to(device)
            recon_batch = sig(model(data))
            recon_loss = loss_function(recon_batch, data)
            try:
                data,coords = next(valPosIter)
            except:
                valPosIter = iter(valPosLoader)
                data,coords = next(valPosIter)
            data = data.to(device)
            recon_batch = sig(model(data))
            count = 0.
            comp_pos_loss = 0.
            comp_neg_loss = 0.
            for i in range(data.shape[0]):
                pos_coords = set()
                instance_count = 0.
                for (z,y,x) in coords[i]:
                    if z == -1:
                        break
                    pos_coords.add((z,y,x))
                    comp_pos_loss -= loss_function(data[i,0,z,y,x],recon_batch[i,0,z,y,x])
                    instance_count += 1
                count += instance_count
                j = 0
                while j < instance_count:
                    z = np.random.randint(0,16)
                    y = np.random.randint(0,512)
                    x = np.random.randint(0,512)
                    if (z,y,x) in pos_coords:
                        continue
                    else:
                        j += 1
                        comp_neg_loss += loss_function(data[i,0,z,y,x],recon_batch[i,0,z,y,x])
            comp_pos_loss /= float(count)
            comp_neg_loss /= float(count)
            print("%d validation loss = %.3f %.3f %.3f"%(epoch,recon_loss,comp_pos_loss,comp_neg_loss))
            validation_loss = recon_loss + (comp_pos_loss + comp_neg_loss) * LAMBDA
        if max_loss == None or training_loss < max_loss:
            max_loss = training_loss
        if (epoch + 1) % 200 == 0:
            torch.save(model.state_dict(),"model/%s/%.3f_%d.pth"%(modelName,validation_loss,epoch))
            
