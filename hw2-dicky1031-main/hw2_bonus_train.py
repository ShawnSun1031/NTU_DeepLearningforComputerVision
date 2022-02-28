# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 13:47:22 2021

@author: dicky1031
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import os
from PIL import Image
import pandas as pd


class hw2_3data(Dataset):
    def __init__(self,root,transform=None):
        self.root = root
        self.transform = transform
        self.filenames = []
        
        #read filenames
        df = pd.read_csv(root+'.csv')
        for i in range(df['image_name'].size):
            filename = os.path.join(root,df['image_name'][i])
            label = df['label'][i]
            self.filenames.append([filename,label])
        self.len = len(self.filenames)

    
    def __getitem__(self,index):
        image_fn,label = self.filenames[index]
        image = Image.open(image_fn)
        # image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image).float()
        
        return image,label
    
    def __len__(self):
        return self.len 


source_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    # transforms.ColorJitter(brightness=0.2,contrast=0.2),
    transforms.ToTensor(),
])
target_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    # transforms.ColorJitter(brightness=0.2,contrast=0.2),
    transforms.ToTensor(),
])

source_dataset = hw2_3data('hw2_data/digits/mnistm/train', transform=source_transform)
target_dataset = hw2_3data('hw2_data/digits/usps/train', transform=target_transform)
test_dataset = hw2_3data('hw2_data/digits/usps/test', transform=target_transform)
print(len(test_dataset))
source_dataloader = DataLoader(source_dataset, batch_size=64, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)







class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
   

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

class DomainClassifier(nn.Module):

    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, h):
        y = self.layer(h)
        return y
    
feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor().cuda()
domain_classifier = DomainClassifier().cuda()
print(feature_extractor)
print(label_predictor)
print(domain_classifier)

feature_extractor.train()
label_predictor.train()
domain_classifier.train()

class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

optimizer_F = optim.Adam(feature_extractor.parameters())
optimizer_C = optim.Adam(label_predictor.parameters())
optimizer_D = optim.Adam(domain_classifier.parameters())
    
# CUDA
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)

# n_batches = len(target_dataset)/64;
# def sample_svhn(step, n_batches):
#     global svhn_set
#     if step % n_batches == 0:
#         svhn_set = iter(target_dataloader)
#     return  svhn_set.next()
   
def train_epoch(source_dataloader, target_dataloader, lamb):
    '''
      Args:
        source_dataloader: source data的dataloader
        target_dataloader: target data的dataloader
        lamb: 調控adversarial的loss係數。
    '''

    # D loss: Domain Classifier的loss
    # F loss: Feature Extrator & Label Predictor的loss
    # total_hit: 計算目前對了幾筆 total_num: 目前經過了幾筆
    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0
    

    for i, ((source_data, source_label),(target_data , _)) in enumerate(zip(source_dataloader,target_dataloader)):
        # print(source_data)
        # print(source_label)
        # print(i)
        # print(len(target_dataloader))
        # print(len(source_dataloader))
        # for r in range(int(len(target_dataloader)/len(source_dataloader))):
        #     if r == 0:
        #         target_data , _ = sample_svhn(r+i*int(len(target_dataloader)/len(source_dataloader)),n_batches)
        #     else: 
        #         target_data_temp , _ = sample_svhn(r+i*int(len(target_dataloader)/len(source_dataloader)),n_batches)
        #         target_data = torch.cat([target_data,target_data_temp],dim=0)
            
        # print(source_data.size())
        # print(target_data.size())
        source_data = source_data.to(device)
        source_label = source_label.to(device)
        target_data = target_data.to(device)
        
        # 我們把source data和target data混在一起，否則batch_norm可能會算錯 (兩邊的data的mean/var不太一樣)
        mixed_data = torch.cat([source_data, target_data], dim=0)
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).to(device)
        # 設定source data的label為1
        domain_label[:source_data.shape[0]] = 1

        # Step 1 : 訓練Domain Classifier
        feature = feature_extractor(mixed_data)
        # 因為我們在Step 1不需要訓練Feature Extractor，所以把feature detach避免loss backprop上去。
        domain_logits = domain_classifier(feature.detach())
        loss = domain_criterion(domain_logits, domain_label)
        running_D_loss+= loss.item()
        loss.backward()
        optimizer_D.step()

        # Step 2 : 訓練Feature Extractor和Label Predictor
        class_logits = label_predictor(feature[:source_data.shape[0]])
        domain_logits = domain_classifier(feature)
        # loss為原本的class CE - lamb * domain BCE，相減的原因同GAN中的Discriminator中的G loss。
        loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
        running_F_loss+= loss.item()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += source_data.shape[0]
        # print(i, end='\r')

    return running_D_loss / (i+1), running_F_loss / (i+1), total_hit / total_num

def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2. / (1+np.exp(-10.*p)) - 1.

# 訓練200 epochs
for epoch in range(50):
    train_D_loss, train_F_loss, train_acc = train_epoch(source_dataloader, target_dataloader, lamb=0.1)
    # torch.save(feature_extractor.state_dict(), f'extractor_model.bin')
    # torch.save(label_predictor.state_dict(), f'predictor_model.bin')

    print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch, train_D_loss, train_F_loss, train_acc))

    label_predictor.eval()
    feature_extractor.eval()
    correct = 0
    for i, (test_data, label) in enumerate(test_dataloader):
        test_data = test_data.cuda()
        label = label.cuda()
    
        class_logits = label_predictor(feature_extractor(test_data))
        pred = class_logits.max(1,keepdim=True)[1]
        correct += pred.eq(label.view_as(pred)).sum().item()
    
        # x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
        # result.append(x)
    label_predictor.train()
    feature_extractor.train()
    
    print('acc: ',correct/len(test_dataset)*100)
    acc = correct/len(test_dataset)*100
    if acc > 80 :
        torch.save(feature_extractor, 'acc_%.2f_extractor_model.pth' %(acc))
        torch.save(label_predictor, 'acc_%.2f_predictor_model.pth'%(acc))

# label_predictor = torch.load('u_to_s_acc_28.48_predictor_model.pth')
# feature_extractor = torch.load('u_to_s_acc_28.48_extractor_model.pth')
# result = []
# label_predictor.eval()
# feature_extractor.eval()

### save csv and calculate acc
# correct = 0
# for i, (test_data, label) in enumerate(test_dataloader):
#     test_data = test_data.cuda()
#     label = label.cuda()
    
#     class_logits = label_predictor(feature_extractor(test_data))
#     pred = class_logits.max(1,keepdim=True)[1]
#     correct += pred.eq(label.view_as(pred)).sum().item()

#     x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
#     result.append(x)
# print('acc: ',correct/len(test_dataset)*100)
# import pandas as pd
# result = np.concatenate(result)

# # Generate your submission
# img_name = []
# for i in range(0,len(result)):
#     img_name.append(str(i).zfill(4)+'.png')
    
# df = pd.DataFrame({'image_name': img_name, 'label': result})
# df.to_csv('m_to_u_DaNN_submission.csv',index=False)

### plot TSNE
#%%
# """Plot TSNE"""
# def plot_with_labels(lowDWeights, labels):
#     plt.cla() #clear当前活动的坐标轴
#     X, Y = lowDWeights[:, 0], lowDWeights[:, 1] #把Tensor的第1列和第2列,也就是TSNE之后的前两个特征提取出来,作为X,Y
#     i = 0
#     x1 = []; y1 = []; c1 = 0
#     x2 = []; y2 = []; c2 = 0
#     x3 = []; y3 = []; c3 = 0
#     x4 = []; y4 = []; c4 = 0
#     x5 = []; y5 = []; c5 = 0
#     x6 = []; y6 = []; c6 = 0
#     x7 = []; y7 = []; c7 = 0
#     x8 = []; y8 = []; c8 = 0
#     x9 = []; y9 = []; c9 = 0
#     x10 = []; y10 = []; c10 = 0
#     for x, y, s in zip(X, Y, labels):
        
#         c = cm.rainbow(int(255 * s /10));
#         if s == 0:
#             x1.append(x)
#             y1.append(y)
#             c1 = c;
#         elif s == 1:
#             x2.append(x)
#             y2.append(y)
#             c2 = c;
#         elif s == 2:
#             x3.append(x)
#             y3.append(y)
#             c3 = c;
#         elif s == 3:
#             x4.append(x)
#             y4.append(y)
#             c4 = c;
#         elif s == 4:
#             x5.append(x)
#             y5.append(y)
#             c5 = c;
#         elif s == 5:
#             x6.append(x)
#             y6.append(y)
#             c6 = c;
#         elif s == 6:
#             x7.append(x)
#             y7.append(y)
#             c7 = c;
#         elif s == 7:
#             x8.append(x)
#             y8.append(y)
#             c8 = c;
#         elif s == 8:
#             x9.append(x)
#             y9.append(y)
#             c9 = c;
#         elif s == 9:
#             x10.append(x)
#             y10.append(y)
#             c10 = c;
#         # #plt.text(x, y, s, backgroundcolor=c, fontsize=9)
#         # if i%50==0:
#         #     plt.text(x, y, str(s),fontdict={'weight': 'bold', 'size':9}) #在指定位置放置文本
#         # plt.scatter(x, y, color=c,label=s)
#         i += 1
#     plt.scatter(x1, y1, color=c1,label='0')
#     plt.scatter(x2, y2, color=c2,label='1')
#     plt.scatter(x3, y3, color=c3,label='2')
#     plt.scatter(x4, y4, color=c4,label='3')
#     plt.scatter(x5, y5, color=c5,label='4')
#     plt.scatter(x6, y6, color=c6,label='5')
#     plt.scatter(x7, y7, color=c7,label='6')
#     plt.scatter(x8, y8, color=c8,label='7')
#     plt.scatter(x9, y9, color=c9,label='8')
#     plt.scatter(x10, y10, color=c10,label='9')
#     plt.xlim(X.min(), X.max());
#     plt.ylim(Y.min(), Y.max());
#     plt.axis('off')
#     plt.title('Visualize usps to svhn (digits) extracter layer');
#     plt.legend(loc = 'lower right')
#     plt.show();
#     plt.pause(0.01)

# ###plot digit TSNE
# for i, (test_data, label) in enumerate(test_dataloader):
#     test_data = test_data.cuda()
#     label = label.cuda()
#     class_logits = feature_extractor(test_data)
#     # TSNE code 
#     if i == 0:
#         tsne = TSNE(perplexity=10, n_components=2, init='pca', n_iter=5000) 
#         plot_only = 2500   
#         low_dim_embs = tsne.fit_transform(class_logits.cpu().data.numpy()[:plot_only, :])
#         pred = class_logits.max(1,keepdim=True)[1]
#         labels = label.view_as(pred).cpu().numpy()[:plot_only]           
#         plot_with_labels(low_dim_embs, labels)
        
# ###plot domain/source
# def plot_domain_with_labels(lowDWeights, labels):
#     plt.cla() #clear当前活动的坐标轴
#     X, Y = lowDWeights[:, 0], lowDWeights[:, 1] #把Tensor的第1列和第2列,也就是TSNE之后的前两个特征提取出来,作为X,Y
#     i = 0
#     x1 = []
#     y1 = []
#     x2 = []
#     y2 = []
#     for x, y, s in zip(X, Y, labels):
        
#         c = cm.rainbow(int(255 * s /2));
#         #plt.text(x, y, s, backgroundcolor=c, fontsize=9)
#         # if i%50==0:
#         #     plt.text(x, y, str(s),fontdict={'weight': 'bold', 'size':9}) #在指定位置放置文本
#         if s == 0:
#             x1.append(x)
#             y1.append(y)
#         elif s == 1:
#             x2.append(x)
#             y2.append(y)
        
#         i += 1
#     plt.scatter(x1, y1, color='r',label='source')
#     plt.scatter(x2, y2, color='b',label='target')
#     plt.xlim(X.min(), X.max());
#     plt.ylim(Y.min(), Y.max());
#     # plt.axis('off')
#     plt.title('Visualize usps to svhn (domain/source) extracter layer');
#     plt.legend()
#     plt.show();
#     plt.pause(0.01)


# for i, ((source_data, source_label),(test_data, test_label)) in enumerate(zip(source_dataloader,test_dataloader)):
#     test_data = test_data.cuda()
#     # test_label = test_label.cuda()
#     source_data = source_data.cuda()
    
    
#     mixed_data = torch.cat([source_data, test_data], dim=0)
#     source_label = torch.ones([source_data.shape[0], 1]).to(device)
#     test_label = torch.zeros([test_data.shape[0], 1]).to(device)
#     mixed_label = torch.cat([source_label,test_label],dim=0)
#     class_logits = feature_extractor(mixed_data)
#     # TSNE code
#     if i == 0:   
#         tsne = TSNE(perplexity=10, n_components=2, init='pca', n_iter=5000) 
#         plot_only = 2500   
#         low_dim_embs = tsne.fit_transform(class_logits.cpu().data.numpy()[:plot_only, :])
#         pred = class_logits.max(1,keepdim=True)[1]
#         labels = mixed_label.view_as(pred).cpu().numpy()[:plot_only]           
#         plot_domain_with_labels(low_dim_embs, labels)
    
 

    # x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
    # result.append(x)
# print('acc: ',correct/len(test_dataset)*100)
# import pandas as pd
# result = np.concatenate(result)


