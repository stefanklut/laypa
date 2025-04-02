# Using YOLO with Laypa pre- and post-processing

This document describes how to use YOLO with Laypa pre- and post-processing. It will guide you through the process of training a YOLO model, and then using it to return PageXML files with bounding boxes for the detected objects.

## Training a YOLO model
In order to train a model you need the same files as for training a Laypa model. The preprocessing will convert the PageXML files to the COCO format, which is used by YOLO.

To start training a YOLO model use a config file for Laypa, in which you specify the region types you want to detect. Then call the [`train_yolo.py`][train_yolo] script as follows:

```bash
python train_yolo.py --config <path_to_config_file> --train <path_to_train_data> --val <path_to_val_data> --yolo <path_to_yolo_model> 
```

The script will preprocess the data and start training the model. The type of training is dependant on the yolo model. For example, if you use the `yolo11n` model, the training will a detection model.
If you use the `yolo11n-seg` the training will be a segmentation model. Currently we recommend using a `yolo11` model as it is the latest architecture and has the best performance. The conversion code is available for detection and segmentation models. No other type of yolo model (e.g. keypoint) is supported.
The `--train` and `--val` arguments should point to the directories with the training and validation data, respectively. The training data should be images with the corresponding PageXML files in a page directory. The validation data should be the same, but with a different set of images. 
The results will be saved in the `runs` directory. From this directory you can find the best model and the training logs. The best model will be saved in the `best.pt` file. This model can then be used for inference. The training logs will be saved in the `train` directory. You can use these logs to monitor the training process and check the performance of the model.

The rest of the available options are the same as for training a Laypa model. You can find more information in the [Laypa documentation](README.md#training). Though these mainly relate to what regions you want to detect. The augmentation options are using the standard YOLO options.

## Using a trained YOLO model
To use a trained model to detect objects in a document, call the [`inference_yolo.py`][inference_yolo] script as follows:

```bash
python inference_yolo.py --config <path_to_config_file> --yolo <path_to_yolo_model> --input <path_to_input_file> --output <path_to_output_file>
```

The path to the model should be the path to the `.pt` file with the trained model.

The script will preprocess the input file and run the model on it. The detected objects will be saved in the output file in the PageXML format, where the post processing converts the bounding boxes to text regions.

The rest of the available options are the same as for using a Laypa model. You can find more information in the [Laypa documentation](README.md#inference).

[train_yolo]: train_yolo.py
[inference_yolo]: inference_yolo.py