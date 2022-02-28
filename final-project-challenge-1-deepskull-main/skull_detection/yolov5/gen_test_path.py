import numpy as np
import os
import pandas as pd

all_files = []
image_ids = []
root = '/home/ubuntu/dlcv_final/skull/test/'
all_dirs = sorted(os.listdir(root))
for k in all_dirs:
    for i in sorted(os.listdir(root+k)):
        fns = root + k + '/' + i
        image_id = os.path.split(fns)[1]
        image_id = image_id.split('.')[0]
        all_files.append(fns)
        image_ids.append(image_id)

dicts = {"path": all_files, 'id': image_ids}
pd.DataFrame(dicts).to_csv('./test_ids.csv', index=None)