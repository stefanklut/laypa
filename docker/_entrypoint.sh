#!/usr/bin/env bash

# set -ef -o pipefail

set +euo pipefail

__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)" || true
if [ -n "${__conda_setup}" ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/condabin/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        . "/opt/condabin/etc/profile.d/conda.sh"
    fi
fi
unset __conda_setup

# __conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)" || true
# if [ -n "${__conda_setup}" ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/home/stefan/miniconda3/etc/profile.d/conda.sh" ]; then
#         # shellcheck disable=SC1091
#         . "/home/stefan/miniconda3/etc/profile.d/conda.sh"
#     fi
# fi
# unset __conda_setup

source activate "$ENV_NAME" 1>/dev/null
# source activate laypa 1>/dev/null
set -euo pipefail

exec "$@"
