#!/bin/bash

if [[ $( builtin cd "$( dirname ${BASH_SOURCE[0]} )/.."; pwd ) != $( pwd ) ]]; then
    DIR_OF_SCRIPT=$( builtin cd "$( dirname ${BASH_SOURCE[0]} )/.."; pwd )
    echo "Change to laypa base folder ($DIR_OF_SCRIPT)"
    cd $DIR_OF_SCRIPT
fi

mkdir -p /tmp/flask

LAYPA_MAX_QUEUE_SIZE=128 \
LAYPA_MODEL_BASE_PATH="/home/tim/Documents/laypa-models/" \
LAYPA_OUTPUT_BASE_PATH="/tmp/flask/" \
FLASK_DEBUG=true FLASK_APP=api.app.py flask run
