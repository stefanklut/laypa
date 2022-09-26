#!/bin/bash

# if [ -z $1 ]; then echo "first parameter should be the path of pylaia-examples" && exit 1; fi;
# EXAMPLES="$(realpath $1)"

docker rmi docker.layout-analysis

echo "Change to directory of script..."
DIR_OF_SCRIPT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR_OF_SCRIPT

echo "Building docker image..."
docker build --squash --no-cache . -t docker.layout-analysis

docker system prune -f
