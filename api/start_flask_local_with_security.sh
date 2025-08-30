#!/bin/bash

if [[ $( builtin cd "$( dirname ${BASH_SOURCE[0]} )/.."; pwd ) != $( pwd ) ]]; then
    DIR_OF_SCRIPT=$( builtin cd "$( dirname ${BASH_SOURCE[0]} )/.."; pwd )
    echo "Change to laypa base folder ($DIR_OF_SCRIPT)"
    cd $DIR_OF_SCRIPT
fi

LAYPA_MAX_QUEUE_SIZE=128 \
LAYPA_MODEL_BASE_PATH="/home/martijnm/workspace/images/laypa-models" \
LAYPA_OUTPUT_BASE_PATH="/tmp/" \
SECURITY_ENABLED="True" \
API_KEY_USER_JSON_STRING='{"1234": "test user"}' \
FLASK_DEBUG=true FLASK_APP=api.flask_app.py flask run