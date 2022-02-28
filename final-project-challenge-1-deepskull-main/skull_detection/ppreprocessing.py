import numpy as np
import pandas as pd
from PIL import Image
import os

def norm_clip_image(imgs):
    imgs = np.clip(imgs, -200, 1000)
    return imgs

if not os.path.isdir('datasets/train/images'):
    os.mkdir('datasets/train/images')

if not os.path.isdir('datasets/val/images'):
    os.mkdir('datasets/val/images')

if not os.path.isdir('datasets/test'):
    os.mkdir('datasets/test')

if not os.path.isdir('datasets/test/images'):
    os.mkdir('datasets/test/images')

root0 = "datasets/train/images/"
csv0 = "./fracture_train_gt.csv"
root1 = "datasets/train/images/"
csv1 = "./new_val_gt.csv"
root2 = "datasets/val/images/"
csv2 = "./new_val_gt.csv"
root3 = "datasets/test/images/"
csv3 = "./test_ids.csv"
def transfer(root1, path):
    col_list = ["path"]
    df = pd.read_csv(path, usecols=col_list)
    for path in df["path"]:
        image = np.load(path).astype(float)
        # image = norm_clip_image(image)
        image = Image.fromarray(image)
        image = image.convert('RGB')
        fn = os.path.split(path)[1]
        image.save(root1 + fn.split('.')[0] + '.jpg')
    
# transfer(root0, csv0)
# transfer(root1, csv1)
# transfer(root2, csv2)
transfer(root3, csv3)

