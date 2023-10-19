#!/bin/bash

if [[ $( builtin cd "$( dirname ${BASH_SOURCE[0]} )/.."; pwd ) != $( pwd ) ]]; then
    DIR_OF_SCRIPT=$( builtin cd "$( dirname ${BASH_SOURCE[0]} )/.."; pwd )
    echo "Change to laypa base folder ($DIR_OF_SCRIPT)"
    cd $DIR_OF_SCRIPT
fi

export LAYPA_MAX_QUEUE_SIZE=128 \
export LAYPA_MODEL_BASE_PATH="/home/stefan/Documents/models/" \
export LAYPA_OUTPUT_BASE_PATH="/tmp/gunicorn" \

mkdir -p $LAYPA_OUTPUT_BASE_PATH

export GUNICORN_RUN_HOST='0.0.0.0:5000'
export GUNICORN_WORKERS=1
export GUNICORN_THREADS=1
export GUNICORN_ACCESSLOG='-'

python api/gunicorn_app.py