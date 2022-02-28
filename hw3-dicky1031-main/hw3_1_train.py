# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 14:36:14 2021

@author: dicky1031
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
import sys
import cv2

# Load image
# NOTE: Assumes an image `img.jpg` exists in the current directory
"""Load Data"""
class hw3_data(Dataset):
    def __init__(self,root,transform=None):
        self.root = root
        self.transform = transform
        self.images = None
        self.labels = []
        self.filenames = []
        
        #read filenames
        file_list = [file for file in os.listdir(self.root)]
        for name in file_list:
            labels = str.split(name,'_')
            labels = int(labels[0])
            filename = os.path.join(self.root, name)
            self.filenames.append((filename,labels))
        self.len = len(self.filenames)

    
    def __getitem__(self,index):
        image_fn,label = self.filenames[index]
        image = Image.open(image_fn).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image).float()
           
        label = torch.tensor(label)
        
        return image,label
    
    def __len__(self):
        return self.len 

batch_size = 4   
class_num = 37
epoch = 15

root = 'hw3_data/p1_data/train'
trans = transforms.Compose([
                        transforms.Resize([384,384]),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
                        ])
trainset = hw3_data(root='hw3_data/hw3_data/p1_data/train',transform=trans)
trainset_loader = DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=1)
testset = hw3_data(root='hw3_data/hw3_data/p1_data/val',transform=trans)
test_loader = DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=1)

# CUDA
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)


from timm import create_model
import torch.nn.functional as F
model_name = "vit_small_patch16_384"
model = create_model(model_name, pretrained=True,num_classes=37).to(device)

# Load ViT
# from pytorch_pretrained_vit import ViT
# model = ViT('B_16_imagenet1k', pretrained=True,patches=14, num_classes= 37,)
# model.to(device)

# print(model)
        
#%%
"""Train the network"""
def train(model):
    optimizer = optim.SGD(model.parameters(),lr=0.0003,momentum=0.9)
    # scheduler1 = lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    criterion = nn.CrossEntropyLoss()
    model.train()
    iteration = 0
    for ep in range(epoch):
        train_loss = 0
        train_acc = 0
        for batch_idx,(data,target) in enumerate(trainset_loader):
           
            data,target = data.to(device),target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output,target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            pred = torch.max(output,1)[1]
            train_correct = (pred==target).sum()
            train_acc += train_correct.item()
            if iteration % 200 == 0:
                print('Train Epoch:{} [{}/{} ({:.0f}%)]\tloss:{:.6f}'.format(ep,batch_idx*len(data),
                        len(trainset_loader.dataset),100.*batch_idx/len(trainset_loader),loss.item()))
            # if iteration % save_interval == 0 and iteration > 0:
            #     save_checkpoint('batch32-%i.pth' % iteration, model, optimizer)
            
            iteration += 1
        
        acc = 100.*train_acc / (len(trainset_loader.dataset))
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(trainset_loader.dataset)), acc))
        if acc > 90:
           torch.save(model, 'timm_acc_%.2f_model.pth' %(acc))
        test(model)
        model.train()
        # # save the final model
        # save_checkpoint('best_batch32-No.-%i epoch.pth' % iteration, model, optimizer)

def test(model):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for i,(img,target) in enumerate(test_loader):
            img,target = img.to(device), target.to(device)
            output = model(img)
            val_loss += criterion(output,target).item()
            pred = torch.max(output,1)[1]
            val_correct = (pred==target).sum()
            val_acc += val_correct.item()
            
            
    # print('Val Loss: {:.6f}, Acc: {:.6f}'.format(val_loss / (len(test_loader.dataset)), 100.*val_acc / (len(test_loader.dataset))))

 
# train(model)



model = torch.load('hw3_1_model.pth')


img = Image.open('hw3_data/hw3_data/p1_data/val/31_4838.jpg').convert("RGB")

plt.figure()
plt.imshow(img)
img_tensor = trans(img).unsqueeze(0).to(device)



pos_embed = model.pos_embed
print(pos_embed.shape)
# Visualize position embedding similarities.
# One cell shows cos similarity between an embedding and all the other embeddings.
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
fig = plt.figure(figsize=(8, 8))
fig.suptitle("Visualization of position embedding similarities", fontsize=24)
for i in range(1, pos_embed.shape[1]):
    sim = F.cosine_similarity(pos_embed[0, i:i+1], pos_embed[0, 1:], dim=1)
    sim = sim.reshape((24, 24)).detach().cpu().numpy()
    ax = fig.add_subplot(24, 24, i)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(sim)




patches = model.patch_embed(img_tensor)  # patch embedding convolution
print("Image tensor: ", img_tensor.shape)
print("Patch embeddings: ", patches.shape)

transformer_input = torch.cat((model.cls_token, patches), dim=1) + pos_embed
print("Transformer input: ", transformer_input.shape)


print("Transformer Multi-head Attention block:")
attention = model.blocks[11].attn
print(attention)
# fc layer to expand the dimension
transformer_input_expanded = attention.qkv(transformer_input)[0]
print("expanded to: ", transformer_input_expanded.shape)
# Split qkv into mulitple q, k, and v vectors for multi-head attantion
qkv = transformer_input_expanded.reshape(577, 3, 12, 32)  # (N=197, (qkv), H=12, D/H=64)
print("split qkv : ", qkv.shape)
q = qkv[:, 0].permute(1, 0, 2)  # (H=12, N=197, D/H=64)
k = qkv[:, 1].permute(1, 0, 2)  # (H=12, N=197, D/H=64)
kT = k.permute(0, 2, 1)  # (H=12, D/H=64, N=197)
print("transposed ks: ", kT.shape)
# Attention Matrix

attention_matrix = q @ kT[:,:,1:]
print("attention matrix: ", attention_matrix.shape)
plt.imshow(attention_matrix[3].detach().cpu().numpy())
# Visualize attention matrix

fig = plt.figure(figsize=(16, 8))
fig.suptitle("Visualization of Attention", fontsize=24)
fig.add_axes()

img = np.asarray(img)
ax = fig.add_subplot(1, 2, 1)
img = cv2.resize(img,[384,384])
ax.imshow(img)
mean_attention_matrix = torch.mean(attention_matrix,dim=0,keepdim=True)
attn_heatmap = mean_attention_matrix[0,0,:].reshape((24,24)).detach().cpu().numpy()
ax = fig.add_subplot(1, 2, 2)
ax.imshow(attn_heatmap)


big_attn_heatmap = cv2.resize(attn_heatmap,[384,384])


heatmapshow = None
heatmapshow = cv2.normalize(big_attn_heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
heatmap = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)

# a = np.repeat(big_attn_heatmap[:, :, np.newaxis], 3, axis=2).astype("uint8")
plt.imsave('test.jpg',big_attn_heatmap)
a = Image.open('test.jpg')
a = np.array(a)
plt.imshow(a)
aa = np.array(a)  

# gray_img = (img[:,:,0]+img[:,:,1]+img[:,:,2])/3
# gray_img = gray_img.astype(np.float32)
merge = cv2.addWeighted(img,0.5,heatmap,0.5,2)
fig = plt.figure(figsize=(16, 8))
fig.suptitle("Visualization of Attention", fontsize=24)
fig.add_axes()
ax = fig.add_subplot(1, 2, 1)
ax.axis("off")
ax.imshow(img)
ax = fig.add_subplot(1, 2, 2)
ax.axis("off")
ax.imshow(merge)
# plt.figure()
# plt.axis("off")
# plt.imshow(merge)



