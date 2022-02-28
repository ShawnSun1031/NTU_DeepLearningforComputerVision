# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 20:30:52 2021

@author: user
"""

import random
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
import os
from PIL import Image
import pandas as pd
from torch.autograd import Variable

# Set random seed for reproducibility
manualSeed = 223
print("Random Seed: ", manualSeed)
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Load Data
class hw2data(Dataset):
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
        if self.transform is not None:
            image = self.transform(image).float()
        
        return image,label
    
    def __len__(self):
        return self.len     

# Attributes
batch_size = 150
image_size = 28
G_out_D_in = 3
G_in = 100
G_hidden = 64
D_hidden = 64
num_class = 10
epochs = 300
lr = 0.0002
beta1 = 0.5

# load data
root = 'hw2_data/digits/mnistm/train'
trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
trainset = hw2data(root='hw2_data/digits/mnistm/train',transform=trans)
dataLoader = DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=1)

# CUDA
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(inputSize, 32, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),


            nn.Conv2d(32, hiddenSize, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hiddenSize),
            nn.LeakyReLU(0.2, inplace=True),


            nn.Conv2d(hiddenSize, hiddenSize*2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hiddenSize*2),
            nn.LeakyReLU(0.2, inplace=True),
  
            nn.Conv2d(hiddenSize*2, hiddenSize*4, 3, 1, 0, bias=False))
        
         # The height and width of downsampled image
        ds_size = image_size // 2 ** 3 - 1

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(hiddenSize*4 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(hiddenSize*4* ds_size ** 2, num_class))
        

    def forward(self, img):
        # print(img.size()) #batch*3*28*28
        out = self.main(img)
        # print(out.size()) # batch*512*2*2
        out = out.view(out.shape[0], -1) 
        # print(out.size()) #batch*2048
        validity = self.adv_layer(out)
        # print(validity.size()) #batch*1
        label = self.aux_layer(out) #batch*10
        # print(label.size())

        return validity, label
    

# Generator
class Generator(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_class, G_in)  
        self.init_size = 8  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(G_in, 128 * self.init_size ** 2))
        
        self.main = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
          
            nn.ConvTranspose2d(128, hiddenSize*4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hiddenSize*4),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.5),

            nn.ConvTranspose2d(hiddenSize*4, hiddenSize*2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hiddenSize*2),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.5),

            nn.ConvTranspose2d(hiddenSize*2, hiddenSize, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hiddenSize),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.5),

            nn.ConvTranspose2d(hiddenSize, outputSize, 4, 1, 2, bias=False),
            nn.Tanh())

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        # print('gen: ',gen_input.size()) #batch*Gin
        out = self.l1(gen_input) 
        # print('out1',out.size()) #batch*128*4**2
        out = out.view(out.shape[0], 128, self.init_size, self.init_size) 
        # print('out2',out.size()) #batch*128*4*4
        img = self.main(out) 
        # print('mani',img.size()) #batch*3*28*28
        return img
    
# Weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
# Train
def train(batch_size):
    # Loss functions
    adversarial_loss = torch.nn.BCELoss()
    auxiliary_loss = torch.nn.CrossEntropyLoss()
    
    # Initialize generator and discriminator
    generator = Generator(G_in, G_hidden, G_out_D_in)
    discriminator = Discriminator(G_out_D_in, D_hidden)
    print(generator)
    print(discriminator)
    
    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)
    auxiliary_loss.to(device)
    
    generator.train()
    discriminator.train()
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
        
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr,betas=(beta1,0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr,betas=(beta1,0.999))  
    
    FloatTensor = torch.cuda.FloatTensor 
    LongTensor = torch.cuda.LongTensor 

    gen_acc = []
    G_losses = []
    D_losses = []

    for epoch in range(epochs):
        
        for i, (imgs, labels) in enumerate(dataLoader):    
            batch_size = imgs.shape[0]

            # Adversarial ground truths

            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            label = Variable(labels.type(LongTensor))
            
            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, G_in))))
            gen_labels = Variable(LongTensor(np.random.randint(0, num_class, batch_size)))
            # Generate a batch of images 
            gen_imgs = generator(z, gen_labels)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
    
            # Loss for real images
            real_pred, real_aux = discriminator(real_imgs)
            d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, label)) /2
            # d_real_loss.backward()
        
            # Loss for fake images
            fake_pred, fake_aux = discriminator(gen_imgs.detach())
            d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) /2
            # d_fake_loss.backward()
    
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss)/2 
            d_loss.backward()
            optimizer_D.step()
            
            # -----------------
            #  Train Generator
            # -----------------
    
            optimizer_G.zero_grad()
            # Loss measures generator's ability to fool the discriminator
            validity, pred_label = discriminator(gen_imgs)
            g_loss =  (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels)) /2
            g_loss.backward()
            optimizer_G.step()
            
            # Output training stats
            if i % 200 == 0:
                saveimg(generator,'hw2_2_img')
                generator.train()
                acc = test_acc()
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %.2f%%] [G loss: %f]"
                      % (epoch, epochs, i, len(dataLoader), d_loss.item(), acc, g_loss.item()))    
                if acc > 80:  
                    torch.save(generator, '%d_%d %d_%d %.2f netG.pkl'% (epoch, epochs, i, len(dataLoader),acc))

        # Save Losses for plotting later
        acc = test_acc()
        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())
        gen_acc.append(acc)
    
    plotImage(G_losses,D_losses,gen_acc)
    torch.save(generator, '%d_%d %d_%d netG.pkl'% (epoch, epochs, i, len(dataLoader)))

# Plot
def plotImage(G_losses, D_losses,acc):
    print('Start to plot!!')
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.title("Generator images accuracy by Classifier")
    plt.plot(acc, label="G")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

def saveimg(netG,save_file_dir):
    # Set random seed for reproducibility
    manualSeed = 223
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    netG.eval()
    noise = torch.tensor(np.random.normal(0, 1, (1000, G_in))).to(device).float()
    list_label = []
    for i in range(10):
        for j in range(100):
            list_label.append(i) 
    label = torch.tensor(list_label).to(device)
    fake = netG(noise,label)
    for i in range(10):
        for j in range(100):
            vutils.save_image(fake[100*i+j], 'hw2_2_img' +'/'+str(i)+'_'+str(j).zfill(0)+'.png', normalize=True)
 

#%% "acc cal"
import digit_classifier as dc
from digit_classifier import  load_checkpoint
import glob

# load digit classifier
net = dc.Classifier()
path = "Classifier.pth"
load_checkpoint(path, net)
if torch.cuda.is_available():
    net = net.to(device)
class hw22data(Dataset):
    def __init__(self,root,transform=None):
        self.root = root
        self.transform = transform
        self.images = None
        self.filenames = []
        self.labels = []
        for i in range(10):
            for j in range(100):
                filename = glob.glob(os.path.join(root,str(i)+'_'+str(j)+'.png'))
                for fn in filename:
                    self.filenames.append((fn,i))           
        self.len = len(self.filenames) 
    def __getitem__(self,index):
        image_fn,label = self.filenames[index]
        image = Image.open(image_fn)

        if self.transform is not None:
            image = self.transform(image).float()
            
        return image,label
    
    def __len__(self):
        return self.len  
genroot = 'hw2_2_img'
trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
genset = hw22data(root=genroot,transform=trans)
genLoader = DataLoader(genset,batch_size=32,shuffle=False,num_workers=1)

def test_acc():
    correct = 0
    for i,(data,label) in enumerate(genLoader):
        data,label = data.to(device),label.to(device)
        output = net(data)
        pred = output.max(1,keepdim=True)[1]
        correct += pred.eq(label.view_as(pred)).sum().item()
    acc = correct/len(genset)*100
    
    return acc      
    
train(batch_size)    

#%%
# def plot100(netG):
#     # Set random seed for reproducibility
#     manualSeed = 223
#     random.seed(manualSeed)
#     torch.manual_seed(manualSeed)
#     np.random.seed(manualSeed)
#     netG.eval()
#     noise = torch.tensor(np.random.normal(0, 1, (100, G_in))).to(device).float()
#     list_label = []
#     img_list = []
#     for i in range(10):
#         for j in range(10):
#             list_label.append(j) 
#     label = torch.tensor(list_label).to(device)
#     fake = netG(noise,label).detach().cpu()
#     img_list.append(vutils.make_grid(fake,10, padding=2, normalize=True))
#     # Plot the fake images from the last epoch
#     plt.figure(figsize=(10, 10))

#     plt.axis("off")
#     plt.title("Gnerate Images")
#     plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
#     plt.show()
    
      
# netG = torch.load('317_500 0_400 92.50 netG.pkl')
# saveimg(netG,'hw2_2_img')
# a = test_acc()
# plot100(netG)














    