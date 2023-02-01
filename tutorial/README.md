# Tutorial
This is a tutorial on how to use Laypa with concrete examples. 

## Setup
First ensure that the proper [install instructions][setup_section] have been followed.
Open a terminal with the Laypa home directory as the current working directory

```sh
cd <PATH_TO_LAYPA>
```

And make sure that the Laypa conda environment is currently active

```sh
conda activate laypa
```

## Training
We are first going to train a model using the data in the [`tutorial/data/training`][training_link] directory. The general approach is described in the main [README][training_section]. The data is structured as follows:

```
training
├── page
│   ├── NL-HaNA_1.01.02_62_0109.xml
│   ├── NL-HaNA_1.01.02_62_0118.xml
│   ├── NL-HaNA_1.01.02_62_0258.jpg
│   └── ...
├── NL-HaNA_1.01.02_62_0109.jpg
├── NL-HaNA_1.01.02_62_0118.jpg
├── NL-HaNA_1.01.02_62_0258.jpg
└── ...
```

Any training data should follow this structure. And the the pageXML should contain (depending on what type of training you have) either baselines and/or regions.

The results (by default) will be saved in the output dir specified in the config file. This includes the trained models at various steps throughout the training process. Additionally, it contains the models that achieved the highest scores in a number of metrics, such as Intersection over Union (IoU) and Accuracy (Acc). Apart from saved models, it also contains all the necessary info to retrain the model from scratch. This includes the full config and a `.txt` file containing all the images used as training and validation data.

During training you should see the loss go down indicating that the model is learning. For the best results more data and a longer training time is usually required. This training is done as an example of how a training loop should function

### Baseline Models

To train the baseline model, we will use the [baseline config][baseline_config_link] found in the tutorial directory. This config inherits most information from the [larger baseline config][baseline_base_link], but overwrites (among other things) the save location of the trained model and the number of iterations.

```sh
python main.py --config tutorial/baseline_tutorial_config.yaml --train tutorial/data/train --val tutorial/data/validation
```

### Region Models

To train the region model, we will use the [region config][region_config_link] found in the tutorial directory. This config inherits most information from the [larger region config][region_base_link], but overwrites (among other things) the save location of the trained model and the number of iterations.

```sh
python main.py --config tutorial/region_tutorial_config.yaml --train tutorial/data/train --val tutorial/data/validation
```

### Altering Config

To change the save location, the `--opts` argument can be used with the `OUTPUT_DIR` argument as follows:
```sh
python main.py --config tutorial/region_tutorial_config.yaml --train tutorial/data/train --val tutorial/data/validation --opts OUTPUT_DIR tutorial/other_results
```

All other aspects of the training can be changed in either the config file directly or using the `--opts` argument to change them from the command line.



## Inference
We are then going to inference the data found in the [`tutorial/data/inference`][inference_link] directory. The general approach is described in the main [README][inference_section]. The data is structured as follows:

```
inference
├── NL-HaNA_1.01.02_3112_0395.jpg
├── NL-HaNA_1.01.02_3124_0022.jpg
└── ...
```

### Baseline Models
For the inference of the baseline, we will again use the [baseline config][baseline_config_link] found in the tutorial directory.

```sh
python run.py --config tutorial/baseline_tutorial_config.yaml --input tutorial/data/inference --output tutorial/inference_results
```
The baseline models output the mask image for further processing. The pageXML output is just the name/placeholder.

### Region Models
For the inference of the regions, we will again use the [region config][region_config_link] found in the tutorial directory.

```sh
python run.py --config tutorial/baseline_tutorial_config.yaml --input tutorial/data/inference --output tutorial/inference_results
```

The region models output the pageXML with regions directly. The pageXML will already contain polygons which indicate where each text region is and what class they belong to.

### Altering Config
To change the weights that are loaded into the model the `--opts` argument can be used with the `TEST.WEIGHTS` argument. These weights should be trained using the same model structure, but perhaps with other hyperparameters, otherwise the weights will not match between models.

```sh
python run.py --config tutorial/baseline_tutorial_config.yaml --input tutorial/data/train --output tutorial/inference_results --opts TEST.WEIGHTS <TRAINING_RUN>/checkpoints/<MODEL_NAME>.pth
```

<!-- Images and Links Shorthand-->
[setup_section]: ../README.md#setup
[training_section]: ../README.md#training
[inference_section]:../README.md#inference

[baseline_config_link]:baseline_tutorial_config.yaml
[region_config_link]:region_tutorial_config.yaml

[baseline_base_link]:../configs/segmentation/region/region_dataset.yaml
[region_base_link]:../configs/segmentation/region/region_dataset.yaml

[training_link]: data/training/
[inference_link]:data/inference/
