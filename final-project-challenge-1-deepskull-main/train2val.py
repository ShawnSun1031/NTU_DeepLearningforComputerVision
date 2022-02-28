# 把自己切的validation set算完後拿到的.csv 拿去跟val_gt.csv比對
# 因為For_student.py 的gt,prediction要對齊，所以這邊生一個對齊好的val_gt.csv出來
# e.g.
# gt_val_csv = "val_gt.csv"
# train2val("train_gt.csv","output_val.csv",gt_val_csv)
# For_student.py的assert 全部刪掉，然後把line115的split改成兩格
import csv
def train2val(gt_train_csv, out_val_csv, gt_val_csv):
    gt = {}
    val_id_list = []
    with open(out_val_csv, newline='') as f:
        rows = csv.reader(f, delimiter=',')
        next(rows)
        for row in rows:
            val_id_list.append(row[0])

    with open(gt_train_csv, newline='') as f:
        rows = csv.reader(f)
        next(rows)
        for row in rows:
            gt[row[0]] = row
    with open(gt_val_csv,'w') as f:
        f.write('id,label,coords\n')
        for val_id in val_id_list:
            f.write( f"{gt[val_id][0]},{gt[val_id][1]},{gt[val_id][2]}\n")

