#!/bin/bash

if !(docker -v &> /dev/null); then
    echo "Docker is not installed please follow https://docs.docker.com/engine/install/"
    exit 1
fi

if !(docker image inspect docker.loghi-tooling:latest &> /dev/null); then
    echo "Loghi tooling is not installed please follow https://github.com/MMaas3/dockerize-images to install"
    exit 1
fi

if !(docker image inspect docker.laypa:latest &> /dev/null); then
    echo "Laypa is not installed please follow https://github.com/MMaas3/dockerize-images to install"
    exit 1
fi

tmpdir=$(mktemp -d)
image_dir=$_arg_input
output_dir$_arg_output

image_dir=/home/stefan/Documents/test
output_dir=/home/stefan/Documents/test2

GPU=$_arg_gpu

DOCKERGPUPARAMS=""
if [[ $GPU -gt -1 ]]; then
        DOCKERGPUPARAMS="--gpus all"
        echo "using GPU"
fi

docker run $DOCKERGPUPARAMS --rm -m 32000m -ti -v image_dir:image_dir docker.laypa:latest \
    python run.py \
    -c configs/segmentation/pagexml_baseline_dataset.yaml \
    -i $image_dir \
    -o $output_dir \
    -m baseline \
    # > /dev/null

if [[ $? -ne 0 ]]; then 
    echo "Baseline detection has errored, stopping program"
fi

docker run $DOCKERGPUPARAMS --rm -m 32000m -ti -v image_dir:image_dir docker.laypa:latest \
    python run.py \
    -c /home/stefan/Documents/repos/laypa/configs/segmentation/pagexml_baseline_dataset_imagenet_freeze.yaml \
    -i $image_dir \
    -o $output_dir \
    -m region \
    # > /dev/null

if [[ $? -ne 0 ]]; then 
    echo "Region detection has errored, stopping program"
fi

docker run --rm -v $SRC/:$SRC/ docker.loghi-tooling /src/loghi-tooling/minions/target/appassembler/bin/MinionExtractBaselines \
    -input_path_png $output_dir/page/ \
    -input_path_page $output_dir/page/ \
    -output_path_page $output_dir/page/

if [[ $? -ne 0 ]]; then 
    echo "Extract baselines has errored, stopping program"
fi
