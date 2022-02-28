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
delim = ''
delims = ' '
root = "/home/ubuntu/dlcv_final/skull/train/"
train_idx, val_idx = np.load("/home/ubuntu/final-project-challenge-1-deepskull/split.npy", allow_pickle=True)
i = 0
for each in json_data["datainfo"]:
    if json_data["datainfo"][str(each)]["path"].split('/')[0] in train_idx:
        tmp = json_data["datainfo"][str(each)]["coords"]
        # if int(json_data["datainfo"][str(each)]["label"]) != 0:
        if tmp == []:
            tmp = ''
            train_coords.append(tmp)
            train_labels.append(json_data["datainfo"][str(each)]["label"])
            train_id.append(root + json_data["datainfo"][str(each)]["path"])
        else:
            k = 0
            tmp = str(tmp)
            tmp = tmp.replace("[", delim)
            tmp = tmp.replace("]", delim)
            count = len(tmp.split(', ')) / 2
            while k < int(count):
                idx = 2*k
                entries = tmp.split(', ')[idx:idx+2]
                x = int(entries[0])
                y = int(entries[1])
                train_coords.append(f'{x-10}, {y-10}, {x+10}, {y+10}')
                train_labels.append(json_data["datainfo"][str(each)]["label"])
                train_id.append(root + json_data["datainfo"][str(each)]["path"])
                k += 1
        # train_coords.append(json_data["datainfo"][str(each)]["coords"])
    else:
        # if int(json_data["datainfo"][str(each)]["label"]) != 0:
        tmp = json_data["datainfo"][str(each)]["coords"]
        if tmp == []:
            tmp = ''
            val_coords.append(tmp)
            val_labels.append(json_data["datainfo"][str(each)]["label"])
            val_id.append(root + json_data["datainfo"][str(each)]["path"])
        else:
            k = 0
            tmp = str(tmp)
            tmp = tmp.replace("[", delim)
            tmp = tmp.replace("]", delim)
            count = len(tmp.split(', ')) / 2
            while k < int(count):
                idx = 2*k
                entries = tmp.split(', ')[idx:idx+2]
                x = int(entries[0])
                y = int(entries[1])
                val_coords.append(f'{x-10}, {y-10}, {x+10}, {y+10}')
                val_labels.append(json_data["datainfo"][str(each)]["label"])
                val_id.append(root + json_data["datainfo"][str(each)]["path"])
                k += 1
        # val_coords.append(json_data["datainfo"][str(each)]["coords"])
def _format(all_coords):
    i = 0
    for unit in all_coords:
        if unit == '':
            all_coords[i] = ''
        else:
            all_coords[i] = str(all_coords[i])
            all_coords[i] = all_coords[i].replace("[", delim)
            all_coords[i] = all_coords[i].replace("]", delim)
            all_coords[i] = all_coords[i].replace("'", delim)
        # all_coords[i] = all_coords[i].replace(",", delims)
        i += 1
    return all_coords

train_coords = _format(train_coords)
val_coords = _format(val_coords)
def label(all_coords):
    k = 0
    fracture_list = []
    label_list = []
    for f in all_coords:
        if all_coords[k] == '':
            fracture_list.append('')
            label_list.append('')
        else:
            fracture_list.append('positive')
            label_list.append('0')
        k += 1
    return fracture_list, label_list

train_fracture_list, train_label_list = label(train_coords)
val_fracture_list, val_label_list = label(val_coords)
# print(len(train_label_list))
# print(len(train_fracture_list))
def make_csv(id, coords, fracture_list, save_path):
    coords = pd.Series(coords).str.split(', ', expand=True)
    # dicts = {"id":id, "coords":coords, "class_name":fracture_list}
    # pd.DataFrame(dicts).to_csv(save_path, index=0)
    dict1 = {"id":id}
    # dict2 = {"idx":[i for i in range(len(id))]}
    dict3 = {"class_name":fracture_list}
    df1 = pd.DataFrame(dict1)
    # df3 = pd.DataFrame(dict2)
    df2 = pd.DataFrame(dict3)
    df = pd.concat([df1, coords, df2], axis=1)
    df.to_csv(save_path, index=0)
# make_csv(train_id, train_coords, train_fracture_list, '/home/ubuntu/dlcv_final/Faster-RCNN/train_gt.csv')
make_csv(val_id, val_coords, val_fracture_list, './val.csv')
def make_cls_id_csv(fracture_list, ids, save_path):
    dicts = {"class_name":fracture_list, "ids":ids}
    pd.DataFrame(dicts).to_csv(save_path, index=0)

# make_cls_id_csv(train_fracture_list, train_label_list, '/home/ubuntu/dlcv_final/train_class_names.csv')
# make_cls_id_csv(val_fracture_list, val_label_list, '/home/ubuntu/dlcv_final/val_class_name.csv')
    