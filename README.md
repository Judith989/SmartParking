# SmartParking
This repository gives a guided tutorial on the Object detection of Vehicles, Pedestrians and Traffic sign based on the [TraPedesVeh dataset](https://github.com/Judith989/TraPedesVeh-A-mini-Dataset-for-Intelligent-Transportation-Systems/blob/main/README.md), and for the paper titled "State-of-the-Art Object Detectors for Vehicle, Pedestrian, and Traffic Sign Detection for Smart Parking Systems", which was presented at the International Conference on Information and Communication Technology Convergence (ICTC2022), Jeju, South Korea.

<img src="https://github.com/Judith989/SmartParking/blob/main/Figures.jpg" width="944">

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
