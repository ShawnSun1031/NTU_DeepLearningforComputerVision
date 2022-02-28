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
import cv2
import sys
model_name = sys.argv[1]

def read_yolov5(kind="test"):
    dic = torch.load("./test.pth")
    t = 0
    out = "id,label,coords\n"
    for key in sorted(list(dic.keys())):
        out += "%s,%s,"%(key.replace(".jpg",""),"1")
        for xywh,c in dic[key]:
            x,y,w,h = xywh
            if c > t:
                out += "%s %s "%(int(x),int(y))
        out += "\n"
    with open(f"pred/"+str(model_name)+"_%s.csv"%(kind),"w") as f:
        f.write(out)
# read_yolov5("train")
read_yolov5("test")

def merge(kind):
    df_cls=pd.read_csv("pred/clf_test_3.csv")
    df_coord=pd.read_csv(f"pred/{model_name}_test.csv")
    out = "id,label,coords\n"
    for row_cls,row_coord in zip(df_cls.iterrows(),df_coord.iterrows()):
        if row_cls[1]["label"] == 0:
            out+="%s,0,\n"%(row_cls[1]["id"])
        else:
            if str(row_coord[1]["coords"])=="nan":
                out+="%s,-1,\n"%(row_cls[1]["id"])
            else:
                out+="%s,1,%s\n"%(row_cls[1]["id"],row_coord[1]["coords"])
    with open(f"pred/{model_name}_merge_test.csv","w") as f:
        f.write(out)
# merge("train")
merge("test")