# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 13:03:57 2021

@author: dicky1031
"""

import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import glob
import os
import numpy as np
from PIL import Image
from torch.optim import lr_scheduler
import pandas as pd
import imageio


def read_masks(filepath):
    '''
    Read masks from directory and tranform to categorical
    '''
    masks = np.empty((512, 512))
    mask = imageio.imread(filepath)
    mask = (mask >= 128).astype(int)
    mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]  
    masks[mask == 3] = 0  # (Cyan: 011) Urban land 
    masks[mask == 6] = 1  # (Yellow: 110) Agriculture land 
    masks[mask == 5] = 2  # (Purple: 101) Rangeland 
    masks[mask == 2] = 3  # (Green: 010) Forest land 
    masks[mask == 1] = 4  # (Blue: 001) Water 
    masks[mask == 7] = 5  # (White: 111) Barren land 
    masks[mask == 0] = 6  # (Black: 000) Unknown
    masks[mask == 4] = 6  # (Red: 100) Unknown

    return masks

def return_mask(mask):
    maskimg = np.zeros((512,512,3)).astype(int)
    maskimg[mask == 1 ,0] = 255
    maskimg[mask == 2 ,0] = 255
    maskimg[mask == 5 ,0] = 255
    
    maskimg[mask == 0 ,1] = 255
    maskimg[mask == 1 ,1] = 255
    maskimg[mask == 3 ,1] = 255
    maskimg[mask == 5 ,1] = 255
    
    maskimg[mask == 0 ,2] = 255
    maskimg[mask == 2 ,2] = 255
    maskimg[mask == 4 ,2] = 255
    maskimg[mask == 5 ,2] = 255
    return maskimg
"""Load Data"""
   
class hw1data(Dataset):
    def __init__(self , root , transform = None):
        self.root = root
        self.transform = transform
        self.images = None
        self.labels = None
        self.sat_filenames = []
        # self.mask_filenames = []
        for filename in sorted(glob.glob(self.root + "/*.jpg" )):
            self.sat_filenames.append(filename)
            # m_filename = filename.replace("sat.jpg","mask.png")
            # self.mask_filenames.append(m_filename)

        self.length = len(self.sat_filenames)
    
    def __getitem__(self,index):
        sat_image_path = self.sat_filenames[index]
        # mask_image_path = self.mask_filenames[index]
        sat_image = imageio.imread(sat_image_path)
        # mask_image = read_masks(mask_image_path)
     


        if self.transform is not None:
            sat_image = self.transform(sat_image).float()
            # mask_image = self.transform(mask_image).long()
        return sat_image 
    
    def __len__(self):
        return self.length    


#%%
"""Use GPU"""
use_cuda = torch.cuda.is_available()
torch.manual_seed(1)
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device used:',device)
#%%
# 使用预训练的 VGG16

# class fcn32s(nn.Module):
#     def __init__(self, num_classes=7):
#         super(fcn32s, self).__init__()
#         #卷积层使用VGG16的
#         pretrained_net = models.vgg16(pretrained=True)
#         self.features = pretrained_net.features
#         #将全连接层替换成卷积层
#         self.classifier = nn.Sequential(
#             nn.Conv2d(512, 4096, kernel_size=(2, 2), stride=(1, 1)),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(0.4),
            
#             nn.Conv2d(4096, 1000, kernel_size=(1, 1), stride=(1, 1)),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(0.5),
            
#             nn.ConvTranspose2d(1000, num_classes, kernel_size=(64, 64) ,stride=(32,32) , padding=0, bias=False),
#         )
#     def  forward (self, x) :        
#         x = self.features(x)
#         x = self.classifier(x)
#         return x
# net = fcn32s().to(device)
# print(net)

class fcn8s(nn.Module):
    def __init__(self, num_classes=7, pretrained = True):
        super(fcn8s, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.to_pool3 = nn.Sequential(*list(vgg.features.children())[:17])
        self.to_pool4 = nn.Sequential(*list(vgg.features.children())[17:23])
        self.to_pool4_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False,return_indices=True ))
        self.to_pool5 = nn.Sequential(*list(vgg.features.children())[24:])  
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=(2, 2), stride=(1, 1))
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout2d(0.4)
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=(1, 1), stride=(1, 1))
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout2d(0.5)
        
        self.up4x = nn.ConvTranspose2d(4096, 256, 8 , 4 , 0, bias=False)
        self.pool4_2xup = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.pool4_2xup_cont = nn.ConvTranspose2d(512, 256, 1 , 1 , 0, bias=False)
        self.up8x = nn.ConvTranspose2d(256, num_classes, 16 , 8 , 4, bias=False)



    def  forward (self, x) :
        pool3_output = self.to_pool3(x) # [64, 256, 32, 32]
        pool4_output = self.to_pool4(pool3_output) #pool4 output size torch.Size([64, 512, 16, 16])
        pool4_1_output,i = self.to_pool4_1(pool4_output)
        pool4_2x = self.pool4_2xup(pool4_1_output,i) # 2x pool4 torch.Size([64, 512, 32, 32])
        pool4_2x_cont = self.pool4_2xup_cont(pool4_2x)
        x = self.to_pool5(pool4_1_output)
        x = self.fc6(x)
        x = self.relu1(x)
        x = self.drop1(x)
        fc7 = self.fc7(x)
        x = self.relu2(fc7)
        x = self.drop2(x)
        fc7_4x = self.up4x(fc7)
        # print(pool3_output.shape)
        # print(fc7.shape,fc7.dtype)
        # print(fc7_4x.shape)
        # print(pool4_2x.shape)
        # print(pool4_2x_cont.shape)
        x = self.up8x(fc7_4x+pool4_2x_cont+pool3_output)
        return x
net = fcn8s().to(device)
print(net)

#%%
def pixel_acc(pred, target):
    correct = np.sum(pred == target)
    total   = np.sum(target == target)
    return correct / total

def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    ious = []
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        # print(tp_fp , tp_fn,  tp)
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        ious.append(iou)
    #     print('class #%d : %1.5f'%(i, iou))
    # print('\nmean_iou: %f\n' % mean_iou)
    #     print('class #%d : %1.5f'%(i, iou))
    # print('\nmean_iou: %f\n' % mean_iou)
    return np.nanmean(ious)

#%%
# """Train the network"""
# def train(model,epoch,save_interval,log_interval=50):
#     optimizer = optim.Adam(model.parameters(), lr=0.0002)
#     # scheduler1 = lr_scheduler.ReduceLROnPlateau(optimizer,'min')
#     criterion = nn.CrossEntropyLoss()
#     model.train()
#     iteration = 0
#     for ep in range(epoch):
#         train_loss = 0
#         mean_iou = 0
#         p_acc = 0
#         for batch_idx,(data,target) in enumerate(train_loader):      
#             data,target = data.to(device),target.to(device)
#             target = target[:,0,:,:]
#             optimizer.zero_grad()
#             output= model(data)
#             loss = criterion(output,target)
#             train_loss += loss.item()
            
#             loss.backward()
#             optimizer.step()
            
#             label_pred = output.data.max(1)[1].data.cpu().numpy() #max(1) : find max in axis = 1 ; [1] after max turn the value to position
#             label_true = target.data.cpu().numpy()
#             mean_iou += mean_iou_score(label_pred, label_true)
#             for pred , labels in zip(label_pred , label_true):
#                 # pred = pred[0,:,:]
#                 p_acc += pixel_acc(pred, labels)
#                 # mean_iou += mean_iou_score(pred, labels)
                 

#             if iteration % log_interval == 0:
#                 print('Train Epoch:{} [{}/{} ({:.0f}%)]\tloss:{:.6f}'.format(ep,batch_idx*len(data),
#                         len(train_loader.dataset),100.*batch_idx/len(train_loader),loss.item()))
#             # if iteration % save_interval == 0 and iteration > 0:
#             #     save_checkpoint('batch32-%i.pth' % iteration, model, optimizer)
            
#             iteration += 1
#         # scheduler1.step(train_loss)
#         P_ACC= 100.* p_acc / len(train_loader.dataset)
#         print('Train Loss: {:.6f}, mean_iou: {:.6f}, p-acc: {:.6f} '.format(train_loss / (batch_idx+1), 
#                                                             100.*mean_iou / (batch_idx+1),
#                                                             P_ACC))

#         test(model)
#         model.train()
#         # save the final model
#         save_checkpoint('FCN32s -No.-%i epoch.pth' % ep, model, optimizer)
        
def test(model,save_file_dir):
    
    criterion = nn.CrossEntropyLoss()
    model.eval()
    val_loss = 0
    # mean_iou = 0
    p_acc = 0
    count = 0
    with torch.no_grad():
        for step,data in enumerate(test_loader):
            # target = target[:,0,:,:]
            # data,target = data.to(device),target.to(device)
            data = data.to(device)
            # target = transforms.Resize([512])(target)
            output = model(data)
            # output = transforms.Resize([512])(output)
            #

            # loss = criterion(output,target).item()
            # val_loss += loss
            # pred = output.max(1,keepdim=True)[1]
            label_pred = output.data.max(1)[1].data.cpu().numpy() #[8,512,512]
            # a.append(label_pred)
            

            # true_labels = target.data.cpu().numpy()   #(batch,512,512)
            # c.append(true_labels)
            # print('pred size: ',pred)
            # mean_iou += mean_iou_score(label_pred, true_labels)

            for pred in label_pred:
                # pred = pred[0,:,:]
                # pred = np.array(resize(pred,output_shape=(512,512),preserve_range=True,order=0))
                # a.append(labels)
                # p_acc += pixel_acc(pred, labels)
                # mean_iou += mean_iou_score(pred, labels)
                pred = return_mask(pred).astype(np.uint8)
                imageio.imsave(save_file_dir +'/'+str(count).zfill(4)+".png",pred)
                # misc.imsave(output_dir+"\\"+str(count).zfill(4)+".png",labelimg)
                count+=1
            # correct += pred.eq(target.view_as(pred)).sum().item()


            
    # P_ACC= 100.* p_acc / len(test_loader.dataset)
    # print('Val Loss: {:.6f}, mean_iou: {:.6f}, p_acc: {:.6f}'.format(val_loss / (step+1),
    #                                                       100.*mean_iou / (step+1),P_ACC))

   
    
    # val_loss = val_loss/len(test_loader.dataset)
    # print('\nVal set: Average loss: {:.4f},Accuracy: {}/{} ({:.0f}%)\n'.format(val_loss,correct,
    #       len(test_loader.dataset),100.*correct/len(test_loader.dataset)))
    # return a,c
       
def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)
#%%
if __name__ == '__main__': 
    # optimizer = optim.Adam(net.parameters(), lr=0.0002)
    # load_checkpoint("best_batch32-No.-9500 epoch.pth",net,optimizer)
    # train(net,30,10)
    import sys
    test_img_dir = sys.argv[1]
    save_file_dir = sys.argv[2]

    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    load_checkpoint('hw1_2_model.pth',net,optimizer)
    # a = imageio.imread('hw1_data/p2_data/train/0000_sat.jpg')
    # a = transforms.ToTensor()(a)
    # train_data = hw1data(root='hw1_data/p2_data/train',transform=transforms.ToTensor())
    test_data = hw1data(root=test_img_dir,transform=transforms.ToTensor())
    # train_loader = DataLoader(train_data,batch_size=8,shuffle=True,num_workers=1)
    test_loader = DataLoader(test_data,batch_size=8,shuffle=False,num_workers=1) 
    
    
    test(net,save_file_dir)
    
    # epoch 21 0.6977
    # class #0 : 0.76181
    # class #1 : 0.87914
    # class #2 : 0.30704
    # class #3 : 0.82222
    # class #4 : 0.73756
    # class #5 : 0.67864
    
    # mean_iou: 0.697735

    # epoch 9 0.6624
    # epoch 5 0.6173
    # epoch 1 0.5034
    # epoch 0 0.4404
    # aaa = a[0]
    # ccc = c[0]
    # for i in range(1,len(a)):
    #     ind = i
    #     aa = a[i]
    #     cc = c[i]
    #     aaa = np.concatenate((aa, aaa))
    #     ccc = np.concatenate((cc,ccc))
        
    # abc = mean_iou_score(aaa, ccc)
        
    # labels = a[0]
    # labels = return_mask(labels)
    # imageio.imsave("hw1_data/mask" +"\\"+str(0).zfill(4)+".png",labels)
    
