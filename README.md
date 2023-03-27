# SmartParking
This repository gives a description of the Object detection of Vehicles, Pedestrians and Traffic sign based on the [TraPedesVeh dataset](https://github.com/Judith989/TraPedesVeh-A-mini-Dataset-for-Intelligent-Transportation-Systems/blob/main/README.md), and for the paper titled "State-of-the-Art Object Detectors for Vehicle, Pedestrian, and Traffic Sign Detection for Smart Parking Systems", which was presented at the International Conference on Information and Communication Technology Convergence (ICTC2022), Jeju, South Korea.

<img src="https://github.com/Judith989/SmartParking/blob/main/Figures.jpg" width="944">

The following tutorials were used as a guide in executing this project: [Tensorflow guide](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#preparing-the-workspace) and [Neptune guide](https://neptune.ai/blog/how-to-train-your-own-object-detector-using-tensorflow-object-detection-api).

Six different models were implemented in this project and the command line instructions for training, evaluation, model exporting and testing have been included in this repository.

### Data

The dataset used in this project was obtained from the [TraPedesVeh dataset](https://github.com/Judith989/TraPedesVeh-A-mini-Dataset-for-Intelligent-Transportation-Systems/blob/main/README.md) repository. This repository contains both the raw image files and the converted TF.record files. For ease of implementataion, you can just use the Tf.record files.



### Training

1. python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config --num_train_steps=1500

2. python model_main_tf2.py --model_dir=models/my_ssd_mobilenet_v2_320x320 --pipeline_config_path=models/my_ssd_mobilenet_v2_320x320/pipeline.config --num_train_steps=1500

3. python model_main_tf2.py --model_dir=models/my_ssd_mobilenet_v1_fpn_640x640 --pipeline_config_path=models/my_ssd_mobilenet_v1_fpn_640x640/pipeline.config --num_train_steps=1500

4. python model_main_tf2.py --model_dir=models/my_faster_rcnn_resnet101_v1_640x640 --pipeline_config_path=models/my_faster_rcnn_resnet101_v1_640x640/pipeline.config --num_train_steps=1500

5. python model_main_tf2.py --model_dir=models/my_faster_rcnn_resnet50_v1_640x640 --pipeline_config_path=models/my_faster_rcnn_resnet50_v1_640x640/pipeline.config --num_train_steps=1500

6. python model_main_tf2.py --model_dir=models/my_faster_rcnn_inception_resnet_v2_640x640 --pipeline_config_path=models/my_faster_rcnn_inception_resnet_v2_640x640/pipeline.config --num_train_steps=1500




### Evaluation

1. python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config --checkpoint_dir=models/my_ssd_resnet50_v1_fpn

2. python model_main_tf2.py --model_dir=models/my_ssd_mobilenet_v2_320x320 --pipeline_config_path=models/my_ssd_mobilenet_v2_320x320/pipeline.config --checkpoint_dir=models/my_ssd_mobilenet_v2_320x320

3.   python model_main_tf2.py --model_dir=models/my_ssd_mobilenet_v1_fpn_640x640 --pipeline_config_path=models/my_ssd_mobilenet_v1_fpn_640x640/pipeline.config --checkpoint_dir=models/my_ssd_mobilenet_v1_fpn_640x640

4.  python model_main_tf2.py --model_dir=models/my_faster_rcnn_resnet101_v1_640x640 --pipeline_config_path=models/my_faster_rcnn_resnet101_v1_640x640/pipeline.config --checkpoint_dir=models/my_faster_rcnn_resnet101_v1_640x640

5. python model_main_tf2.py --model_dir=models/my_faster_rcnn_resnet50_v1_640x640 --pipeline_config_path=models/my_faster_rcnn_resnet50_v1_640x640/pipeline.config --checkpoint_dir=models/my_faster_rcnn_resnet50_v1_640x640

6. python model_main_tf2.py --model_dir=models/my_faster_rcnn_inception_resnet_v2_640x640 --pipeline_config_path=models/my_faster_rcnn_inception_resnet_v2_640x640/pipeline.config --checkpoint_dir=models/my_faster_rcnn_inception_resnet_v2_640x640

### Model Export
1. python exporter_main_v2.py     --input_type="image_tensor"    --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config   --trained_checkpoint_dir=models/my_ssd_resnet50_v1_fpn/    --output_directory=exported-models/my_ssd_resnet50_v1_fpn

2. python exporter_main_v2.py     --input_type="image_tensor"    --pipeline_config_path=models/my_ssd_mobilenet_v2_320x320/pipeline.config   --trained_checkpoint_dir=models/my_ssd_mobilenet_v2_320x320/    --output_directory=exported-models/my_ssd_mobilenet_v2_320x320

3. python exporter_main_v2.py     --input_type="image_tensor"    --pipeline_config_path=models/my_ssd_mobilenet_v1_fpn_640x640/pipeline.config   --trained_checkpoint_dir=models/my_ssd_mobilenet_v1_fpn_640x640/    --output_directory=exported-models/my_ssd_mobilenet_v1_fpn_640x640

4. python exporter_main_v2.py     --input_type="image_tensor"    --pipeline_config_path=models/my_faster_rcnn_resnet101_v1_640x640/pipeline.config   --trained_checkpoint_dir=models/my_faster_rcnn_resnet101_v1_640x640/    --output_directory=exported-models/my_faster_rcnn_resnet101_v1_640x640

5. python exporter_main_v2.py     --input_type="image_tensor"    --pipeline_config_path=models/my_faster_rcnn_resnet50_v1_640x640/pipeline.config   --trained_checkpoint_dir=models/my_faster_rcnn_resnet50_v1_640x640/    --output_directory=exported-models/my_faster_rcnn_resnet50_v1_640x640

6. python exporter_main_v2.py     --input_type="image_tensor"    --pipeline_config_path=models/my_faster_rcnn_inception_resnet_v2_640x640/pipeline.config   --trained_checkpoint_dir=models/my_faster_rcnn_inception_resnet_v2_640x640/    --output_directory=exported-models/my_faster_rcnn_inception_resnet_v2_640x640


### Testing

1. python detect_objects.py --video_input --class_ids "7" --threshold 0.3  --video_path annotations/testvid.mp4 --model_path exported-models/my_ssd_resnet50_v1_fpn/saved_model --path_to_labelmap annotations/label_map.pbtxt


If you experience any issues while running the test, kindly refer to this [Github](https://github.com/opencv/opencv/issues/8537) solution, and make the following commands in Linux:
* $ conda remove opencv
* $ conda install -c menpo opencv
* $ pip install --upgrade pip
* $ pip install opencv-contrib-python

### Abstract

To meet the safety requirements of pedestrians and other vehicles in smart parking systems, vehicles rely heavily on visual data to classify and detect target objects. In real-time, deep learning (DL)-based algorithms have shown excellent results in object detection. While several studies have thoroughly investigated various DL-based object detection methods, only a few have focused on multi-detection, encompassing pedestrian detection, vehicle detection, and traffic sign detection. This work provided a comparative analysis of six independent variants of Faster-RCNN and SSD Models for pedestrians, vehicles, and traffic sign detection. There is a lack of comparison leveraging on detailed evaluation metrics among existing models. To address the issue of little or non-availability of datasets for multi-detection, this work provided a mini dataset for task (TraPedesVeh). Experimental results analyzed using detection accuracy, average precision (AP), average recall (AR), training time, and loss show that the Faster-RCNN model with the Inception backbone outperformed all other models.


### Paper
[State-of-the-Art Object Detectors for Vehicle, Pedestrian, and Traffic Sign Detection for Smart Parking Systems](https://ieeexplore.ieee.org/document/9952856).

### Authors
Judith Nkechinyere Njoku∗, Goodness Oluchi Anyanwu†, Ikechi Saviour Igboanusi†, Cosmas Ifeanyi Nwakanma∗, and Dong-Seong Kim†

### Citation
If you use this work in your research, please cite this [paper](https://ieeexplore.ieee.org/document/9952856).


### Requirements

* pip install tfslim
* pip install tensorflow==2.* 
* pip install scipy
* pip install tensorflow-io
* pip install matplotlib
* pip install tf-models-official
* pip install opencv-contrib-python

