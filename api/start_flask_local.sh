#!/bin/bash

if [[ $( builtin cd "$( dirname ${BASH_SOURCE[0]} )/.."; pwd ) != $( pwd ) ]]; then
    DIR_OF_SCRIPT=$( builtin cd "$( dirname ${BASH_SOURCE[0]} )/.."; pwd )
    echo "Change to laypa base folder ($DIR_OF_SCRIPT)"
    cd $DIR_OF_SCRIPT
fi

LAYPA_MAX_QUEUE_SIZE=128 \
LAYPA_MODEL_BASE_PATH="/home/stefan/Documents/models/" \
LAYPA_OUTPUT_BASE_PATH="/tmp/" \
FLASK_DEBUG=true FLASK_APP=api.flask_app.py flask run