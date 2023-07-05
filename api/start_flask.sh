#!/bin/bash

if [[ $( builtin cd "$( dirname ${BASH_SOURCE[0]} )/.."; pwd ) != $( pwd ) ]]; then
    DIR_OF_SCRIPT=$( builtin cd "$( dirname ${BASH_SOURCE[0]} )/.."; pwd )
    echo "Change to laypa base folder ($DIR_OF_SCRIPT)"
    cd $DIR_OF_SCRIPT
fi

FLASK_DEBUG=true FLASK_APP=api.flask_app.py flask run