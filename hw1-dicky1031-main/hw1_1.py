# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:46:14 2021

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
import matplotlib.pyplot as plt
from matplotlib import cm 
from sklearn.manifold import TSNE
import pandas as pd
#%%
# Seed
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


"""Use GPU"""
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device used:',device)

"""Load Data"""
class hw1data(Dataset):
    def __init__(self,root,transform=None):
        self.root = root
        self.transform = transform
        self.images = None
        self.labels = []
        self.filenames = []
        
        #read filenames
        file_list = [file for file in os.listdir(self.root)]
        for name in file_list:
            label = name
            filename = os.path.join(self.root, name)
            self.filenames.append((filename,label))
        self.len = len(self.filenames)

    
    def __getitem__(self,index):
        image_fn,label = self.filenames[index]
        image = Image.open(image_fn)
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image,label
    
    def __len__(self):
        return self.len


#%%
"""Pretrained VGG16 model"""
class VGGNet(nn.Module):
    def __init__(self, num_classes=50):
        super(VGGNet, self).__init__()
        net = models.vgg16_bn(pretrained=True)
        net.classifier = nn.Sequential()
        self.features = net
        self.fc1 = nn.Linear(512*7*7,4096)
        self.relu1 = nn.ReLU(True)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU(True)
        self.dropout2 = nn.Dropout(0.3)
        self.fc5 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        lastlayer = x
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc5(x)
        return x,lastlayer
    
model = VGGNet().to(device)  #use GPU
print(model)

#%%
# """Plot TSNE"""
# def plot_with_labels(lowDWeights, labels):
#     plt.cla() #clear当前活动的坐标轴
#     X, Y = lowDWeights[:, 0], lowDWeights[:, 1] #把Tensor的第1列和第2列,也就是TSNE之后的前两个特征提取出来,作为X,Y
#     i = 0
#     for x, y, s in zip(X, Y, labels):
        
#         c = cm.rainbow(int(255 * s /50));
#         #plt.text(x, y, s, backgroundcolor=c, fontsize=9)
#         if i%50==0:
#             plt.text(x, y, str(s),fontdict={'weight': 'bold', 'size':9}) #在指定位置放置文本
#         plt.scatter(x, y, color=c,label=s)
#         i += 1
#     plt.xlim(X.min(), X.max());
#     plt.ylim(Y.min(), Y.max());
#     plt.title('Visualize last layer');
#     plt.show();
#     plt.pause(0.01)
#%%
# """Train the network"""
# def train(model,epoch,save_interval,log_interval=200):
#     optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.9,weight_decay=0.0001)
#     # scheduler1 = lr_scheduler.ReduceLROnPlateau(optimizer,'min')
#     criterion = nn.CrossEntropyLoss()
#     model.train()
#     iteration = 0
#     for ep in range(epoch):
#         train_loss = 0
#         train_acc = 0
#         for batch_idx,(data,target) in enumerate(trainset_loader):
           
#             data,target = data.to(device),target.to(device)
#             optimizer.zero_grad()
#             output,_ = model(data)
#             loss = criterion(output,target)
#             train_loss += loss.item()
#             loss.backward()
#             optimizer.step()
#             pred = torch.max(output,1)[1]
#             train_correct = (pred==target).sum()
#             train_acc += train_correct.item()
#             if iteration % log_interval == 0:
#                 print('Train Epoch:{} [{}/{} ({:.0f}%)]\tloss:{:.6f}'.format(ep,batch_idx*len(data),
#                         len(trainset_loader.dataset),100.*batch_idx/len(trainset_loader),loss.item()))
#             # if iteration % save_interval == 0 and iteration > 0:
#             #     save_checkpoint('batch32-%i.pth' % iteration, model, optimizer)
            
#             iteration += 1

#         print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(trainset_loader.dataset)), 100.*train_acc / (len(trainset_loader.dataset))))
#         test(model)
#         model.train()
#         # save the final model
#         save_checkpoint('best_batch32-No.-%i epoch.pth' % iteration, model, optimizer)

#%%
def test(model,savepath):
    
    # criterion = nn.CrossEntropyLoss()
    model.eval()
    # val_loss = 0
    # correct = 0
    pred_label = []
    image_id = []
    with torch.no_grad():
        for step,(data,target) in enumerate(test_loader):
            data,target = data.to(device),target
            output,last_layer = model(data)
            # val_loss += criterion(output,target).item()
            pred = output.max(1,keepdim=True)[1]
            # correct += pred.eq(target.view_as(pred)).sum().item()

            #               TSNE code
            # if step % 100 == 0:   
            #     tsne = TSNE(perplexity=10, n_components=2, init='pca', n_iter=5000) 
            #     plot_only = 2500   
            #     low_dim_embs = tsne.fit_transform(last_layer.cpu().data.numpy()[:plot_only, :])
            #     labels = target.view_as(pred).cpu().numpy()[:plot_only]           
            #     plot_with_labels(low_dim_embs, labels)
            
            for img_id , im_pred in zip(target , pred):
                image_id.append(img_id)
                pred_label.append(int(im_pred.cpu().numpy()))
 
    dicts = {"image_id" :image_id , "label" : pred_label }
    DF = pd.DataFrame(dicts)
    DF.to_csv(savepath,index = 0)    
    
    # val_loss = val_loss/len(test_loader.dataset)
    # print('\nVal set: Average loss: {:.4f},Accuracy: {}/{} ({:.0f}%)\n'.format(val_loss,correct,
    #       len(test_loader.dataset),100.*correct/len(test_loader.dataset)))


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
    
    optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.9,weight_decay=0.0001)
    load_checkpoint('hw1_1_model.pth', model, optimizer)
    import sys
    test_img_dir = sys.argv[1]
    save_file_dir = sys.argv[2]
    testset_path = test_img_dir 
    savepath = save_file_dir

    
    
    trans = transforms.Compose([
                            transforms.Resize([64,64]),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
                            ])
    testset = hw1data(root=test_img_dir,transform=trans)
    # print('# images in valset:',len(testset))
    test_loader = DataLoader(testset,batch_size=32,shuffle=False,num_workers=1) 
    test(model,savepath)
