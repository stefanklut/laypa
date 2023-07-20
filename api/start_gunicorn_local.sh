#!/bin/bash

if [[ $( builtin cd "$( dirname ${BASH_SOURCE[0]} )/.."; pwd ) != $( pwd ) ]]; then
    DIR_OF_SCRIPT=$( builtin cd "$( dirname ${BASH_SOURCE[0]} )/.."; pwd )
    echo "Change to laypa base folder ($DIR_OF_SCRIPT)"
    cd $DIR_OF_SCRIPT
fi

LAYPA_MAX_QUEUE_SIZE=16 \
LAYPA_MODEL_BASE_PATH="/home/stefan/Documents/models/" \
LAYPA_OUTPUT_BASE_PATH="/tmp/" \

GUNICORN_RUN_HOST='0.0.0.0:5000'
GUNICORN_WORKERS=1
GUNICORN_THREADS=1
GUNICORN_ACCESSLOG='-'

python api/gunicorn_app.py