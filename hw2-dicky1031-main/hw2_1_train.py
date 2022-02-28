# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

# Set random seed for reproducibility
manualSeed = 123
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(inputSize, hiddenSize, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hiddenSize, hiddenSize*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hiddenSize*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hiddenSize*2, hiddenSize*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hiddenSize*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hiddenSize*4, hiddenSize*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hiddenSize*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hiddenSize*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid())

    def forward(self, input):
        return self.main(input)
    


# Generator
class Generator(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(inputSize, hiddenSize*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hiddenSize*8),
            nn.ReLU(inplace=True),


            nn.ConvTranspose2d(hiddenSize*8, hiddenSize*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hiddenSize*4),
            nn.ReLU(inplace=True),


            nn.ConvTranspose2d(hiddenSize*4, hiddenSize*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hiddenSize*2),
            nn.ReLU(inplace=True),


            nn.ConvTranspose2d(hiddenSize*2, hiddenSize, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hiddenSize),
            nn.ReLU(inplace=True),


            nn.ConvTranspose2d(hiddenSize, outputSize, 4, 2, 1, bias=False),
            nn.Tanh())

    def forward(self, input):
        return self.main(input)
    
    
    
    
# CUDA
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)


"""Load Data"""
class hw2data(Dataset):
    def __init__(self,root,transform=None):
        self.root = root
        self.transform = transform
        self.images = None
        self.filenames = []
        
        #read filenames
        file_list = [file for file in os.listdir(self.root)]
        for name in file_list:
            filename = os.path.join(self.root, name)
            self.filenames.append(filename)
        self.len = len(self.filenames)

    
    def __getitem__(self,index):
        image_fn = self.filenames[index]
        image = Image.open(image_fn)
        
        if self.transform is not None:
            image = self.transform(image).float()
        
        return image
    
    def __len__(self):
        return self.len  


# Attributes

batch_size = 128
image_size = 64
G_out_D_in = 3
G_in = 100
G_hidden = 64
D_hidden = 64

epochs = 1000
lr = 0.0002
beta1 = 0.5

root = 'hw2_data/face/train'
trans = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
                        ])
trainset = hw2data(root='hw2_data/face/train',transform=trans)
dataLoader = DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=1)
testset = hw2data(root='hw2_data/face/test',transform=trans)
test_loader = DataLoader(testset,batch_size=32,shuffle=False,num_workers=1)



# Weights
def weights_init(m):
    classname = m.__class__.__name__
    print('classname:', classname)

    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
        
# Train
def train():
    # Create the generator
    netG = Generator(G_in, G_hidden, G_out_D_in).to(device)
    netG.apply(weights_init)
    print(netG)

    # Create the discriminator
    netD = Discriminator(G_out_D_in, D_hidden).to(device)
    netD.apply(weights_init)
    print(netD)

    # Loss fuG_out_D_intion
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, G_in, 1, 1, device=device)

    real_label = 1
    fake_label = 0
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    print('Start!')
    netD.train()
    netG.train()

    for epoch in range(epochs):

        for i, data in enumerate(dataLoader, 0):

            # Update D network
            netD.zero_grad()
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device).float()
            output = netD(real_cpu).view(-1)

            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, G_in, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)

            errD_fake = criterion(output, label)
            errD_fake.backward()

            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Update G network
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (epoch, epochs, i, len(dataLoader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                saveimg(netG,'hw2_1_img')
                IS = cal_IS(netG)
                print('Is score: %.4f'%IS)
            netG.train()
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == epochs - 1) and (i == len(dataLoader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()

                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            iters += 1
            if (IS >= 2.15) & (i % 50 == 0):
                # torch.save(netD, 'netD.pkl')
                torch.save(netG, '%d_%d  %d_%d %.2f netG.pkl'% (epoch, epochs, i, len(dataLoader),IS))
    plotImage(G_losses,D_losses,img_list)
    if IS >= 2.15:
        # torch.save(netD, 'netD.pkl')
        torch.save(netG, '%d_%d %d_%d %.2f netG.pkl'% (epoch, epochs, i, len(dataLoader),IS))

    return G_losses, D_losses



# Plot
def plotImage(G_losses, D_losses,img_list):
    print('Start to plot!!')
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataLoader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch.to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()
    
def saveimg(netG,save_file_dir):
    # Set random seed for reproducibility
    manualSeed = 123
    #manualSeed = random.randint(1, 10000) # use if you want new results
    # print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    netG.eval()
    noise = torch.randn(1000, 100, 1, 1, device=device)
    fake = netG(noise)
    count = 0
    for i in range(1000):
        vutils.save_image(fake[i], 'hw2_1_img' +'/'+str(count).zfill(4)+'.png', normalize=True)
        count+=1
        
def showimg(img):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.axis('off')
    plt.title('the first 32 generated images')
    # plt.savefig(path)  
    
    
def cal_IS(netG):
    dataroot = 'hw2_1_img'
    trans = transforms.Compose([
                            transforms.Scale(32),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
                            ])
    g_img = hw2data(root=dataroot,transform=trans)
    IS = inception_score(g_img, cuda=True, batch_size=32, resize=True, splits=10)
    return IS[0]
    
    
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
from scipy.stats import entropy
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
  
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


train() 


# # save 1000 img
# netG = torch.load('hw2_1 2.177 netG.pkl')
# saveimg(netG,'hw2_1_img')
# # save 32 grid
# # Set random seed for reproducibility
# manualSeed = 123
# #manualSeed = random.randint(1, 10000) # use if you want new results
# print("Random Seed: ", manualSeed)
# random.seed(manualSeed)
# torch.manual_seed(manualSeed)
# netG.eval()
# noise = torch.randn(32, 100, 1, 1, device=device)
# fake = netG(noise)
# showimg(vutils.make_grid(fake.cpu(),padding=2, normalize=True))


# FID 27.06  IS 2.21

# import pytorch_fid
# pytorch_fid('/home/md703/Documents/SCS/hw2-dicky1031/hw2_data/face/test','/home/md703/Documents/SCS/hw2-dicky1031/hw2_1_img')



    