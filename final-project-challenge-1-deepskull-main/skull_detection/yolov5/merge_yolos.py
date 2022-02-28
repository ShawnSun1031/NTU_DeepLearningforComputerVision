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

def compute_points(j,coords,t_dis=32):
    if len(coords) % 2 == 0:
        t = len(coords)//2
    else:
        t = len(coords)//2 + 1
        
        
    if len(coords[j]) == 0:
        return []
    else:
        comp = list(range(len(coords)))
        comp.remove(j)
        valid_points = []
        for (x,y) in coords[j]:
            count = 1
            for k in comp:
                for (xc,yc) in coords[k]:
                    if abs(x-xc)+abs(y-yc) < t_dis:
                        count += 1
                        break
            if count >= t:
                valid_points.append((x,y))
        return valid_points

def ps2str(ps):
    s = ""
    for (x,y) in ps:
        s += "%s %s "%(x,y)
    return s

def mergeyolos(*paths,out_path):
    dfs = []
    for path in paths:
        dfs.append(pd.read_csv(path))
    out = "id,label,coords\n"
    for i in range(len(dfs[0]["coords"])):    
        coords = []
        for j in range(len(dfs)):
            if str(dfs[j]["coords"][i])=="nan":
                coords.append([])
            else:
                xys = dfs[j]["coords"][i].split(" ")
                dfj_coords = []
                for l in range(len(xys)//2):
                    dfj_coords.append([int(xys[2*l]),int(xys[2*l+1])])
                coords.append(dfj_coords)
        ps = []
        for j in range(len(coords)):
            ps += compute_points(j,coords)
        if len(ps) == 0:
            out+="%s,-1,\n"%(dfs[0]["id"][i])
        else:
            out+="%s,1,%s\n"%(dfs[0]["id"][i],ps2str(ps))            
    with open("%s"%(out_path),"w") as f:
        f.write(out)
def merge_clf_yolos(kind):
    df_cls=pd.read_csv("pred/clf_test_4.csv")
    df_coord=pd.read_csv("pred/merge_yolos_test.csv")
    out = "id,label,coords\n"
    for row_cls,row_coord in zip(df_cls.iterrows(),df_coord.iterrows()):
        if row_cls[1]["label"] == 0:
            out+="%s,0,\n"%(row_cls[1]["id"])
        else:
            if str(row_coord[1]["coords"])=="nan":
                out+="%s,-1,\n"%(row_cls[1]["id"])
            else:
                out+="%s,1,%s\n"%(row_cls[1]["id"],row_coord[1]["coords"])
    with open("pred/merge_yolos_clf_%s.csv"%(kind),"w") as f:
        f.write(out)
mergeyolos("pred/yolov5s_0.61_merge_test.csv","pred/yolov5s_0.66_merge_test.csv","pred/yolov5x_0.69_merge_test.csv",out_path="pred/merge_yolos_test.csv")
merge_clf_yolos("test")