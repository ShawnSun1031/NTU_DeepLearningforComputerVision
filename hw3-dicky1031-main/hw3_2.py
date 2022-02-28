# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 23:25:23 2021

@author: dicky1031
"""
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch
import cv2
from transformers import BertTokenizer

from models import caption
from datasets import coco, utils
from configuration import Config
import sys

test_file_dir = sys.argv[1]
save_file_dir = sys.argv[2]


for filename in os.listdir(test_file_dir):
    filedir = os.path.join(test_file_dir,filename)
    config = Config()
    model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
    end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
    
    image = Image.open(filedir).convert("RGB")
    image = coco.val_transform(image)
    image = image.unsqueeze(0)
    
    
    def create_caption_and_mask(start_token, max_length):
        caption_template = torch.zeros((1, max_length), dtype=torch.long)
        mask_template = torch.ones((1, max_length), dtype=torch.bool)
    
        caption_template[:, 0] = start_token
        mask_template[:, 0] = False
    
        return caption_template, mask_template
    
    
    caption, cap_mask = create_caption_and_mask(
        start_token, config.max_position_embeddings)
    
    
    @torch.no_grad()
    def evaluate():
        model.eval()
        for i in range(config.max_position_embeddings - 1):
            predictions,pos,weight = model(image, caption, cap_mask)
            predictions = predictions[:, i, :]
            predicted_id = torch.argmax(predictions, axis=-1)
    
            if predicted_id[0] == 102:
                return caption,pos,weight
    
            caption[:, i+1] = predicted_id[0]
            cap_mask[:, i+1] = False
    
        return caption,pos,weight

    
    output,pos,weight = evaluate()
    
    result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)

    
    a = weight[5].squeeze()
    
    img = Image.open(filedir).convert("RGB")
    img = np.asarray(img)
    img = cv2.resize(img,[299,199])

    fig = plt.figure(figsize=(24, 12))
    fig.suptitle("Visualization of Attention", fontsize=24)
    fig.add_axes()
    

    ax = fig.add_subplot(4, 4, 1)
    ax.set_title(tokenizer.decode(output[0][0].tolist(),fontsize=24, skip_special_tokens=True))
    ax.annotate("---->", (300, 100), fontsize=48, annotation_clip=False)
    ax.axis("off")
    ax.imshow(img)
    
    for i in range(15):
        ax = fig.add_subplot(4, 4, i+2)
        aaa = a[i,:].reshape(pos[0].shape[-2],pos[0].shape[-1])
        b = np.asarray(aaa)
        b = cv2.resize(b,[299,199])

        heatmapshow = None
        heatmapshow = cv2.normalize(b, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_RAINBOW)
        merge = cv2.addWeighted(img,0.4,heatmap,0.6,10)
     
        ax.set_title(tokenizer.decode(output[0][i+1].tolist(),fontsize=24, skip_special_tokens=True))
        ax.annotate("---->", (300, 100), fontsize=48, annotation_clip=False)
        ax.axis("off")
        ax.imshow(merge)
    filename = filename.replace('.jpg', '.png')
    plt.savefig(save_file_dir+filename)


