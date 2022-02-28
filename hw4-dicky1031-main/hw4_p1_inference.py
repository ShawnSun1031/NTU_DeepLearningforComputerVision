import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd

from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image

    def __len__(self):
        return len(self.data_df)

class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)

def conv_block(in_channel , out_channel):
    bn = nn.BatchNorm2d(out_channel)
    return nn.Sequential(
        nn.Conv2d(in_channel , out_channel ,3 ,padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
        )

class Convnet(nn.Module):
    def __init__(self , in_channel = 3 , hid_channel = 64 , out_channel = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channel , hid_channel),
            conv_block(hid_channel , hid_channel),
            conv_block(hid_channel , hid_channel),
            conv_block(hid_channel , hid_channel),
            )
    def forward(self,x):
        x = self.encoder(x)
       
        return x.view(x.size(0),-1)


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def cos_metric(a,b):
    n_x = a.shape[0]
    n_y = b.shape[0]
    normalised_x = a / (a.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)
    normalised_y = b / (b.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)

    expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
    expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

    cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
    return cosine_similarities

class distance(nn.Module):
    def __init__(self,out_channel=5,in_channel=5,hid_channel=256):
        super().__init__()
        self.out_channel = out_channel
        self.linear1 = nn.Sequential(
            nn.Linear(in_channel, hid_channel*4),
            nn.ReLU(),
            nn.Linear(hid_channel*4, hid_channel*3),
            nn.ReLU(),
            nn.Linear(hid_channel*3, hid_channel*2),
            nn.ReLU(),
            nn.Linear(hid_channel*2, hid_channel),
            nn.ReLU(),
            nn.Linear(hid_channel, 128),
            nn.ReLU(),
            nn.Linear(128, out_channel),
            )
        
    def forward(self,a,b):
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n,m,-1)
        b = b.unsqueeze(0).expand(n,m,-1)
        x = ((a - b)**2).sum(dim=2)
        x = self.linear1(x)
  
        return x

    
def predict(model, data_loader):
    prediction_results = []
    
    with torch.no_grad():
        # each batch represent one episode (support data + query data)
        for i, data in enumerate(data_loader):

            # split data into support and query data
            support_input = data[:5 * n_shot,:,:,:] 
            query_input   = data[5 * n_shot:,:,:,:]

            # TODO: extract the feature of support and query data
            proto = model(support_input.cuda())
            proto = proto.reshape(n_shot, 5, -1).mean(dim=0)
            
            q_proto = model(query_input.cuda())

            # TODO: calculate the prototype for each class according to its support data
            logits = euclidean_metric(q_proto, proto)
            # TODO: classify the query data depending on the its distense with each prototype
            pred = torch.argmax(logits, dim=1)
            # acc = (pred == query_label).type(torch.cuda.FloatTensor).mean().item()
            prediction_results.append(pred.cpu().numpy())

    return prediction_results

if __name__=='__main__':
    
    test_csv_file = sys.argv[1]
    img_file_dir = sys.argv[2]
    testcase_csv_file = sys.argv[3]
    out_csv_file = sys.argv[4]
    
    n_way = 5
    n_shot = 1
    n_query = 15
    
    test_dataset = MiniDataset(test_csv_file, img_file_dir)

    test_loader = DataLoader(
        test_dataset, batch_size=n_way * (n_query + n_shot),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(testcase_csv_file))

    # TODO: load your model
    
    # checkpoint = torch.load('1-epoch-max-acc=0.4122.pth')
    model = Convnet().cuda()  
    model.load_state_dict(torch.load('hw4_p1_model.pth'))
    model.eval()
    
    # d_model = distance().cuda()
    # d_model.load_state_dict(checkpoint['modelB'])
    # d_model.eval()

    prediction_results = predict(model, test_loader)

    # TODO: output your prediction to csv
    head = ["episode_id"]
    for i in range(n_way*n_query):
        head.append("query"+str(i))
    df = pd.DataFrame(columns=head)
    
    for i in range(len(prediction_results)):
        df.loc[i] = np.insert(prediction_results[i],0,i)
    
    df.to_csv(out_csv_file,index = 0)
    
    
# 30 ways euclidean = Accuracy: 47.34 +- 0.87 %
# 5 ways cosine = Accuracy: 38.89 +- 0.75 %
# 5 ways parameric model = Accuracy: 44.64 +- 0.82%
# 5 ways 1 shots euclidean = Accuracy: 40.00 +- 0.75 %
# 5 ways 5 shots 61.08%
# 5 ways 10 shots 68.54%


