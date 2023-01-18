<!-- TODO Shields (e.g. Licence Issues KNAW) -->

# Laypa
Laypa is a segmentation network, with the goal of finding regions (paragraph, page number, etc.) and baselines in documents. The current approach is using a ResNet backbone and a feature pyramid head, which made pixel wise classifications. The models are built using the [detectron2][detectron_link] framework. The baselines and region classifications are then made available for further processing. This post-processing turn the classification into instances. So that they can be used by other programs (OCR/HTR), either as masks or directly as pageXML.

<!-- TODO Table of contents -->
## Table of Contents
- [Laypa](#laypa)
  - [Table of Contents](#table-of-contents)
  - [Tested Environments](#tested-environments)
  - [Setup](#setup)
    - [Conda](#conda)
    - [Docker](#docker)
    - [Pretrained models](#pretrained-models)
  - [Dataset(s)](#datasets)
  - [Training](#training)
  - [Inference](#inference)
    - [Without External Processing](#without-external-processing)
    - [With External Java Processing](#with-external-java-processing)
  - [Tutorial](#tutorial)
  - [Evaluation](#evaluation)
  - [License](#license)
  - [Contact](#contact)
    - [Issues](#issues)
    - [Contributions](#contributions)

## Tested Environments
Developed using the following software and hardware:

Operating System | Python | PyTorch | Cudatoolkit | GPU | CUDA | CPU | Success
-|-|-|-|-|-|-|-
 Ubuntu 22.04 | 3.10 | 1.13.0 | 11.7 | RTX 3080 Ti Laptop | 12.0 | Intel i9-12900H | :white_check_mark:

<details>
<summary> Click here to show all tested environments </summary>

<!-- TODO Add more with testing -->
More coming soon

Operating System | Python | PyTorch | Cudatoolkit | GPU | CUDA | CPU | Success
-|-|-|-|-|-|-|-
 Ubuntu 22.04 | 3.10 | 1.13.0 | 11.7 | RTX 3080 Ti Laptop | 12.0 | Intel i9-12900H | :white_check_mark:

</details>

## Setup
The recommended way of running Laypa is inside a conda environment. To ensure easier compatibility a method of building a docker is also provided.

To start clone the github repo to your local machine using either HTTPS:
```sh
git clone https://github.com/stefanklut/laypa.git
```

Or using SSH:
```sh
git clone git@github.com:stefanklut/laypa.git
```

And make laypa the working directory:
```sh
cd laypa
```

### Conda
If not already installed, install either conda or miniconda ([install instructions][conda_install_link]), or mamba ([install instructions][mamba_install_link]). 

The required packages are listed in the [`environment.yml`][environment_link] file. The environment can be automatically created using the following commands.

Using conda/miniconda:
```sh
conda env create -f environment.yml
```

Using mamba:
```sh
mamba env create -f environment.yml
```

When running Laypa always activate the conda environment
```sh
conda activate laypa
```

### Docker
If not already installed, install the Docker Engine ([install instructions][docker_install_link]). The docker environment can most easily be build with the provided script.

Copy the docker install scripts and Dockerfile(s) to a temporary directory. This is necessary due to the script having to copy the directory it is in. This is not allowed and thus a different external directory is used as build context.

```sh
# Or other location for a temporary directory
tmpdir=$(mktemp -d)
cp -r docker $tmpdir
cd $tmpdir/docker
```

Building the docker using the provided script:
```sh
./buildImage.sh PATH_TO_LAYPA
```

<details>
<summary> Click for manual docker install instructions </summary>

First copy the Laypa directory to the temporary docker directory:
```sh
cp -r <PARENT_DIR>/laypa $tmpdir/docker
```

Change the working dir to the docker directory:
```sh
cd $tmpdir/docker
```

Build the docker using the buildkit version of docker build
```sh
docker buildx build --no-cache . -t docker.laypa
```

</details>


<details>
<summary> Click for minikube install instructions </summary>

<!-- TODO This -->
Minikube is local Kubernetes, allowing you to test the Laypa tools in a Kubernetes environment. If not already installed start with installing minikube ([install instructions][minikube_install_link])

If the docker images have already been built the minikube can run them straight away. To do so, start minikube without any special arguments:
```sh
minikube start
```

Afterwards the docker for Laypa can be added to the running minikube instance using the following command (assuming the Laypa docker was built under the name docker.laypa):
```sh
minikube image load docker.laypa
```

It is also possible to build the Laypa docker using the minikube docker instance. This means minikube will need access to the Laypa code. As it stand, this is current still done using a copy command from the local storage. In order to do so start the minikube with the mount argument:
```sh
minikube start --mount
```
This will make the machines filesystem available to minikube. Then ssh into the running minikube:
```sh
minikube ssh
```

Within the ssh minikube go to the location of the laypa where the host `/home/<user>` is mounted to `minikube-host`
```sh
cd minikube-host/PATH_TO_LAYPA
```

And follow the instructions for install a docker version of Laypa as described [here](#docker)

</details>

When successful the docker image should be available under the name `docker.laypa`. This can be verified using the following command:
```sh
docker image ls
```
And checking if docker.laypa is present in the list of built images.

### Pretrained models
<!-- TODO Add the pretrained models as a download -->
Coming soon

## Dataset(s)

The dataset used for training requires images combined with ground truth pageXML. For structure the pageXML needs to be inside a directory one level down from the images. The dataset can be split over multiple directories, with the image paths specified in a `.txt` file. The structure should look as follows:
```sh
training_data
├── page
│   ├── image1.xml
│   ├── image2.xml
│   ├── image3.xml
│   └── ...
├── image1.jpg
├── image2.jpg
├── image3.jpg
└── ...
```

Where the image and pageXML filename stems should match `image1.jpg <-> image1.xml`. For the `.txt` based dataset absolute paths to the images are recommended. The structure for the data used as validation is the same as that for training.

When running inference the images you want processed should be in a single directory. With the images directly under the root folder as follows:
```sh
inference_data
├── image1.jpg
├── image2.jpg
├── image3.jpg
└── ...
```

Some dataset that should work with laypa are listed below, some preprocessing may be require:
- [cBAD][cbad_link]
- [VOC and notarial deeds][voc_link]
- [OHG][ohg_link]
- [Bozen][bozen_link]

## Training 
Three things are required to train a model using [`main.py`][main_link].
1. A config file, See [`configs/segmentation`][configs_link] for examples of config files and their contents.
2. Ground truth training/validation data in the form of images and their corresponding pageXML. The training/validation data can be provided by giving either a `.txt` file containing image paths or the path of a directory containing there images.

Required arguments:
```sh
python main.py \
    -c/--config <CONFIG> \
    -t/--train <TRAIN [TRAIN ...]> \ 
    -v/--val <VAL [VAL ...]>
```

<details>
<summary> Click to see all arguments </summary>

Optional arguments:
```sh
python main.py \
    -c/--config CONFIG \
    -t/--train TRAIN [TRAIN ...] \
    -v/--val VAL [VAL ...] \
    [--tmp_dir TMP_DIR] \
    [--keep_tmp_dir] \
    [--num-gpus NUM_GPUS] \
    [--num-machines NUM_MACHINES] \
    [--machine-rank MACHINE_RANK] \
    [--dist-url DIST_URL] \
    [--opts ...]
```

The optional arguments are shown using square brackets. The `--tmp_dir` parameter specifies a folder in which to store temporary files. While the `--keep_tmp_dir` parameter prevents the temporary files from being deleted after a run (mostly for debugging).

The remaining arguments are all for training with multiple GPUs or on multiple nodes. `--num-gpus` specifies the number of GPUs per machine. `--num-machines` specifies the number of nodes in the network. `--machine-rank` gives a node a unique number. `--dist-url` is the URL for the PyTorch distributed backend. The final parameter `--opts` allows you to change values specified in the config files. For example, `--opts SOLVER.IMS_PER_BATCH 8` sets the batch size to 8.
</details>

As indicated by the trailing dots multiple training sets can be passed to the training model at once. This can also be done using the train argument multiple types. The `.txt` files can also be mixed with the directories. For example:
```sh
# Pass multiple directories at once
python main.py -c config.yml -t data/training_dir1 data/training_dir2 -v data/validation_set
# Pass multiple directories with multiple arguments
python main.py -c config.yml -t data/training_dir1 -t data/training_dir2 -v data/validation_set
# Mix training directory with txt file
python main.py -c config.yml -t data/training_dir -t data/training_file.txt -v data/validation_set
```

## Inference
To run the trained model on images without ground truth, the images need to be in a single directory. The output consists of either pageXML in the case of regions or a mask in the other cases. This mask can then be processed using other tools to turn the pixel predictions into valid pageXML (for example on baselines). As stated, the regions are turned into polygons for the pageXML within the program already.

How to run the Laypa inference individually will be explained first, and how to run it with the full scripts that include the conversion from images to pageXML with come after.

### Without External Processing
To just run the Laypa inference in [`run.py`][run_link], you need three things:
1. A config file, See [`configs/segmentation`][configs_link] for examples of config files and their contents.
2. A directory with images to be processed
3. A location to which the processed files can be written. The directory will be created if it does not exist yet.

Required arguments
```sh
python run.py \
    -c/--config CONFIG \ 
    -i/--input INPUT \ 
    -o/--output OUTPUT
```
<details>
<summary> Click to see all arguments </summary>

Optional arguments:
```sh
python run.py \
    -c/--config CONFIG \ 
    -i/--input INPUT \ 
    -o/--output OUTPUT
    [--opts ...]
``` 
The optional arguments are shown using square brackets. The final parameter `--opts` allows you to change values specified in the config files. For example, `--opts SOLVER.IMS_PER_BATCH 8` sets the batch size to 8.
</details>

An example of how to call the `run.py` command is given below:
```sh
python run.py -c config.yml -i data/inference_dir -o results_dir
```


### With External Java Processing
<!-- TODO Remove the need for Java -->
Examples of running the full pipeline (with processing of baselines) are present in the [`scripts`][scripts_link] directory. These files make the assumption that the docker images for both Laypa and the loghi-tooling (Java post-processing) are available on your machine. The script will also try and verify this. The Laypa docker image needs to be build with the pretrained models included.

To run the scripts only two thing are needed:
1. A directory with images to be processed.
2. A location to which the processed files can be written. The directory will be created if it does not exist yet.

Required arguments:
```sh
./scripts/pipeline.sh <input> <output>
```
<details>
<summary> Click to see all arguments </summary>

Optional arguments:
```sh
./scripts/pipeline.sh \
        <input> \
        <output> \ 
        -g/--gpu GPU
``` 
The required arguments are shown using angle brackets. The `--gpu` parameter specifies what GPU(s) is accessible to the docker containers. The default is `all`.
</details>


The positional arguments input and output refer to the input and output directory. An example of running the one of the pipelines is shown below:
```sh
./scripts/pipeline.sh inference_dir results_dir
```


## Tutorial
<!-- TODO Small example for training with images and pretrained network -->
For a small tutorial using some concrete examples see the [`tutorial`][tutorial_link] directory.

## Evaluation
The Laypa repository also contains a few tools used to evaluate the results generated by the model.

The first tool is a visual comparison between the predictions of the model and the ground truth. This is done as an overlay of the classes over the original image. The overlay class names and colors are taken from the dataset catalog. The tool to do this is [`eval.py`][eval_link]. The visualization has almost the same arguments as the training command ([`main.py`][main_link]).

Required arguments:
```sh
python eval.py \
    -c/--config <CONFIG> \
    -t/--train <TRAIN [TRAIN ...]> \ 
    -v/--val <VAL [VAL ...]>
```

<details>
<summary> Click to see all arguments </summary>

Optional arguments:
```sh
python eval.py \
    -c/--config CONFIG \
    -t/--train TRAIN [TRAIN ...] \
    -v/--val VAL [VAL ...] \
    [--tmp_dir TMP_DIR] \
    [--keep_tmp_dir]
    [--opts]
```

The optional arguments are shown using square brackets. The `--tmp_dir` parameter specifies a folder in which to store temporary files. While the `--keep_tmp_dir` parameter prevents the temporary files from being deleted after a run (mostly for debugging). The final parameter `--opts` allows you to change values specified in the config files. For example, `--opts SOLVER.IMS_PER_BATCH 8` sets the batch size to 8.
</details>

Example of running [`eval.py`][eval_link]:

```sh
python eval.py -c config.yml -t training_dir -v validation_dir
```

The [`eval.py`][eval_link] will then open a window with both the prediction and the ground truth side by side. Allowing for easier comparison. The visualization masks are created in the same way the preprocessing converts pageXML to masks.

The second tool is a program to compare the similarity of two sets of pageXML. This can mean either comparing ground truth to predicted pageXML, or determining the similarity of two annotations by different people. This tool is the [`xml_comparison.py`][xml_comparison_link] file. The comparison allows you to specify how regions and baseline should be drawn in when creating the pixel masks. The pixel masks are then compared based on their Intersection over Union (IoU) and Accuracy (Acc) scores. For the sake of the Accuracy metric one of the two sets needs to be specified as the ground truth set. So one set is the ground truth directory (`--gt`) argument and the other is the input directory (`--input`) argument.

Required arguments:
```sh
python xml_comparison.py \ 
    -g/--gt GT [GT ...] \
    -i/--input INPUT [INPUT ...]
```

<details>
<summary> Click to see all arguments </summary>

Optional arguments:
```sh
python xml_comparison.py \ 
    -g/--gt GT [GT ...] \
    -i/--input INPUT [INPUT ...] \
    [-m/--mode {baseline,region,start,end,separator,baseline_separator}] \
    [--regions REGIONS [REGIONS ...]] \
    [--merge_regions [MERGE_REGIONS]] \
    [--region_type REGION_TYPE [REGION_TYPE ...]] \
    [-w/--line_width LINE_WIDTH] \
    [-l/line_color {0-255}]  
```

The optional arguments are shown using square brackets. The `--mode` parameter specifies what type of prediction the model has to do. If the mode is region, the `--regions` argument specifies which regions need to be extracted from the pageXML (for example "page-number"). The `--merge_regions` then specifies if any of these regions need to be merged. This could mean converting "insertion" into "resolution" since they are talking about the same thing `resolution:insertion`. The final region argument is `--region_type` which can specify the region type of a region. In the other modes lines are used. The line arguments are `--line_width`, which specifies the line width, and `--line_color`, which specifies the line color.
</details>

The final tool is a program for showing the pageXML as mask images. This can help with showing how the pageXML regions and baseline look. This can be done in gray scale, color, or as a colored overlay over the original image. This tool is located in the [xml_viewer.py][xml_viewer_link] file. It requires an input directory (`--input`) argument and output directory (`--output`) argument.

Required arguments:
```sh
python xml_comparison.py \ 
    -i/--input INPUT [INPUT ...] \
    -o/--output OUTPUT [OUTPUT ...]
```

<details>
<summary> Click to see all arguments </summary>

Optional arguments:
```sh
python xml_comparison.py \ 
    -g/--gt GT [GT ...] \
    -o/--output OUTPUT [OUTPUT ...] \
    [-m/--mode {baseline,region,start,end,separator,baseline_separator}] \
    [--regions REGIONS [REGIONS ...]] \
    [--merge_regions [MERGE_REGIONS]] \
    [--region_type REGION_TYPE [REGION_TYPE ...]] \
    [-w/--line_width LINE_WIDTH] \
    [-l/line_color {0-255}] \
    [-t/--output_type {gray,color,overlay}]
```

The optional arguments are shown using square brackets. The `--mode` parameter specifies what type of prediction the model has to do. If the mode is region, the `--regions` argument specifies which regions need to be extracted from the pageXML (for example "page-number"). The `--merge_regions` then specifies if any of these regions need to be merged. This could mean converting "insertion" into "resolution" since they are talking about the same thing `resolution:insertion`. The final region argument is `--region_type` which can specify the region type of a region. In the other modes lines are used. The line arguments are `--line_width`, which specifies the line width, and `--line_color`, which specifies the line color. The final argument `--output_type` is used to select an output style as either gray scale, color, or a colored overlay.
</details>

## License
Distributed under the MIT License. See [`LICENSE`][license_link] for more information.

## Contact
This project was made while working at the [KNAW Humanities Cluster Digital Infrastructure][huc_di_link]
### Issues
Please let report any bugs or errors that you find to the [issues][issues_link] page. So that they can be looked into. Try to see if an issue with the same problem/bug is not still open. Feature requests should also be done through the [issues][issues_link] page.

### Contributions
If you discover a bug or missing feature that you would like to help with please feel free to send a [pull request][pull_request_link]. 


<!-- Images and Links Shorthand-->
[detectron_link]: https://github.com/facebookresearch/detectron2
[conda_install_link]: https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation
[mamba_install_link]: https://mamba.readthedocs.io/en/latest/installation.html
[docker_install_link]: https://docs.docker.com/engine/install/
[minikube_install_link]: https://minikube.sigs.k8s.io/docs/start/

[cbad_link]: https://doi.org/10.5281/zenodo.2567397
[voc_link]: https://doi.org/10.5281/zenodo.3517776
[ohg_link]: https://doi.org/10.5281/zenodo.3517776
[bozen_link]: https://doi.org/10.5281/zenodo.218236

<!-- TODO Replace with relative links? -->
[pull_request_link]: https://github.com/stefanklut/laypa/pulls
[issues_link]: https://github.com/stefanklut/laypa/issues
[environment_link]: environment.yml
[license_link]: LICENSE
[configs_link]: configs/segmentation/
[scripts_link]: scripts/
[tutorial_link]: tutorial/
[main_link]: main.py
[run_link]: run.py
[eval_link]: eval.py
[xml_comparison_link]: xml_comparison.py
[xml_viewer_link]: xml_viewer.py

[huc_di_link]: https://di.huc.knaw.nl/