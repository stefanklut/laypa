# Using YOLO with Laypa pre- and post-processing

This document describes how to use YOLO with Laypa pre- and post-processing. It will guide you through the process of training a YOLO model, and then using it to return PageXML files with bounding boxes for the detected objects.

## Training a YOLO model
In order to train a model you need the same files as for training a Laypa model. The preprocessing will convert the PageXML files to the COCO format, which is used by YOLO.

To start training a YOLO model use a config file for Laypa, in which you specify the region types you want to detect. Then call the [`train_yolo.py`][train_yolo] script as follows:

```bash
python train_yolo.py --config <path_to_config_file> --train <path_to_train_data> --val <path_to_val_data>
```

The script will preprocess the data and start training the model. The results will be saved in the `runs` directory.

The rest of the available options are the same as for training a Laypa model. You can find more information in the [Laypa documentation](README.md#training).

## Using a trained YOLO model
To use a trained model to detect objects in a document, call the [`run_yolo.py`][run_yolo] script as follows:

```bash
python run_yolo.py --config <path_to_config_file> --yolo <path_to_model> --input <path_to_input_file> --output <path_to_output_file>
```

The path to the model should be the path to the `.pt` file with the trained model.

The script will preprocess the input file and run the model on it. The detected objects will be saved in the output file in the PageXML format, where the post processing converts the bounding boxes to text regions.

The rest of the available options are the same as for using a Laypa model. You can find more information in the [Laypa documentation](README.md#inference).

[train_yolo]: train_yolo.py
[run_yolo]: run_yolo.py