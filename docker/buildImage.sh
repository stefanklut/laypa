#!/bin/bash

#!/bin/bash

if [ -z $1 ]; then echo "first parameter should be the path of laypa" && exit 1; fi;
LAYPA="$(realpath $1)"

docker rmi docker.laypa

echo "Change to directory of script..."
DIR_OF_SCRIPT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR_OF_SCRIPT

cp -r $LAYPA laypa

echo "Building docker image..."
docker build --squash --no-cache . -t docker.laypa

rm -rf layout
docker system prune -f
