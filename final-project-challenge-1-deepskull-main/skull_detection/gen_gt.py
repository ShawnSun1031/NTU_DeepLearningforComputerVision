import json
import pandas as pd
import numpy as np
f = open("/home/ubuntu/dlcv_final/skull/records_train.json")
json_data = json.load(f)
train_labels = []
val_labels = []
train_coords = []
val_coords = []
train_fracture_list = []
val_fracture_list = []
train_id = []
val_id = []
root = "/home/ubuntu/dlcv_final/skull/train/"
train_idx, val_idx = np.load("/home/ubuntu/final-project-challenge-1-deepskull/split.npy", allow_pickle=True)
for each in json_data["datainfo"]:
    if json_data["datainfo"][str(each)]["path"].split('/')[0] in train_idx:
        train_labels.append(json_data["datainfo"][str(each)]["label"])
        train_coords.append(json_data["datainfo"][str(each)]["coords"])
        train_id.append(root+json_data["datainfo"][str(each)]["path"])
    else:
        val_labels.append(json_data["datainfo"][str(each)]["label"])
        val_coords.append(json_data["datainfo"][str(each)]["coords"])
        val_id.append(root+json_data["datainfo"][str(each)]["path"])

delim = ''
delims = ' '
def _format(all_coords):
    i = 0
    for unit in all_coords:
        if unit == []:
            all_coords[i] = ''
        else:
            all_coords[i] = str(all_coords[i])
            all_coords[i] = all_coords[i].replace("[", delim)
            all_coords[i] = all_coords[i].replace("]", delim)
            all_coords[i] = all_coords[i].replace(",", delims)
            all_coords[i] = all_coords[i].replace("  ", delims)
        i += 1
    return all_coords

train_coords = _format(train_coords)
val_coords = _format(val_coords)

def label(all_coords):
    k = 0
    fracture_list = []
    for f in all_coords:
        if all_coords[k] == '':
            fracture_list.append('')
        else:
            fracture_list.append('positive')
        k += 1
    return fracture_list

train_fracture_list = label(train_coords)
# val_fracture_list = label(val_coords)

def make_csv(id, coords, labels, save_path):
    # coords = pd.Series(coords).str.split(', ', expand=True)
    dicts = {"id":id, "label":labels, "coords":coords}
    pd.DataFrame(dicts).to_csv(save_path, index=0)
    """
    dict1 = {"id":id}
    dict3 = {"class_name":fracture_list}
    df1 = pd.DataFrame(dict1)
    df2 = pd.DataFrame(dict3)
    df = pd.concat([df1, coords, df2], axis=1)
    df.to_csv(save_path, index=0)
    """
# print(train_coords)
# make_csv(train_id, train_coords, train_labels, '/home/ubuntu/dlcv_final/inference_val.csv')
make_csv(val_id, val_coords, val_labels, '/home/ubuntu/dlcv_final/inference_val.csv')