bash get_dataset.sh
cd case_level/ 
bash download_model.sh
python3 clf_pred.py --device cuda:0 --dataset_path ../skull/ --num_models 3
cp pred/clf_test_3.csv  ../skull_detection/yolov5/pred/clf_test_3.csv
python3 clf_pred.py --device cuda:0 --dataset_path ../skull/ --num_models 4
cp pred/clf_test_4.csv  ../skull_detection/yolov5/pred/clf_test_4.csv
cd ../skull_detection
python3 preprocessing.py
bash ./download_models.sh
cd yolov5
python3 detect_test.py --weights ../0.61_yolov5s.pt --source ../datasets/test/images/ --imgsz=512 --nosave
python3 yolov5_eval.py "yolov5s_0.61"
python3 detect_test.py --weights ../0.69_yolov5x.pt --source ../datasets/test/images/ --imgsz=512 --nosave
python3 yolov5_eval.py "yolov5x_0.69"
python3 detect_test.py --weights ../0.66_yolov5s.pt --source ../datasets/test/images/ --imgsz=512 --nosave
python3 yolov5_eval.py "yolov5s_0.66"
python3 merge_yolos.py
echo "to TAs love merge_yolos_clf_test.csv" 
