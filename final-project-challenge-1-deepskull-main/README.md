# DLCV Final Project ( Skull Fractracture Detection )


# How to run your code?
```
bash get_test_csv.sh
```
> ***And you will get our final result "merge_yolos_clf_test.csv"***  
> ***The bash details is under below.***

<details>
  <summary>Bash Details</summary>
 
## Download Dataset

```
bash get_dataset.sh
```
>put skull/ under this folder

 
## Case-Level
1. run
 
    ```
    cd case_level/ 
    ```

2. run 

   download models

   ```
   bash download_model.sh
   ```
   run classification prediction

   ```
   python3 clf_pred.py --device cuda:0 --dataset_path ../skull/ --num_models 3
   cp pred/clf_test_3.csv  ../skull_detection/yolov5/pred/clf_test_3.csv
   ```
   Then, the ensemble case level classification result will be dumped at pred/clf_test_3.csv. This gives our highest case level accuracy 0.869 on the leaderboard.

   check best case level accuracy file pred/clf_test_3.csv is 0.869

3. run

   ```
   python3 clf_pred.py --device cuda:0 --dataset_path ../skull/ --num_models 4
   cp pred/clf_test_4.csv  ../skull_detection/yolov5/pred/clf_test_4.csv
   ```


4. run
    ```
    cd ../skull_detection
    ```

## Centroid-Level Hit Rate 
### Training Part
#### Check if there exist empty folder with filename "images" for each in skull_detection/datasets/train, skull_detection/datasets/val, skull_detection/datasets/test, respectively. If not, create them.
    python3 preprocessing.py
    cd yolov5
    python3 train.py --img 512 --batch -1 --epoch 120 --data datasets.yaml --weights yolov5x.pt
> #### train.py 如果發現100個epoch裡面model沒有任何進步，會自動 early stop
> #### Start Inference Part 如果想要直接在TRAINING PHASE完成後對testing set做inference，再run以下的code
    python3 detect_test.py --weights runs/train/exp/weights/best.pt --source ../datasets/test/images/ --imgsz=512 --nosave
    python3 yolov5_eval.py $1\
>   $1 is your output model name
> #### your output coords will be saved in "pred" folder
> TODO: merge with classification results if you want 
### Inference Part (with downloaded checkpoints)
#### Your current directory should be at /final-project-challenge-1-deepskull/skull_detection/
    python3 preprocessing.py
    bash ./download_models.sh
    cd yolov5
    python3 detect_test.py --weights ../0.61_yolov5s.pt --source ../datasets/test/images/ --imgsz=512 --nosave
    python3 yolov5_eval.py "yolov5s_0.61"
    python3 detect_test.py --weights ../0.69_yolov5x.pt --source ../datasets/test/images/ --imgsz=512 --nosave
    python3 yolov5_eval.py "yolov5x_0.69"
    python3 detect_test.py --weights ../0.66_yolov5s.pt --source ../datasets/test/images/ --imgsz=512 --nosave
    python3 yolov5_eval.py "yolov5s_0.66"
> #### your outputs will be saved in "pred" folder
    python3 merge_yolos.py
> #### your output will be saved in "pred" folder with filename "merge_yolos_clf_test.csv"
> #### run below commands if you want to evaluate
    python3 for_students_eval.py --pred_file pred/merge_yolos_clf_test.csv --gt_file 
### check best f1 is 0.709
### End of Inference Part
</details>
