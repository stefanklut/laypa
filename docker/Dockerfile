FROM condaforge/miniforge3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get dist-upgrade -y && \
    apt-get clean -y && \
    apt-get install -y \
    apt-utils git \
    ninja-build gcc g++ \
    ffmpeg libsm6 libxext6 libegl1 libopengl0 && \
    apt-get clean -y && \
    apt-get autoremove -y

WORKDIR /src/
COPY laypa/ laypa/
COPY _entrypoint.sh /_entrypoint.sh

# When github is open
# RUN git clone https://github.com/stefanklut/laypa.git

WORKDIR /src/laypa
# Removing the nsight-compute folder as it is not needed for runtime
RUN mamba env create -p /env -f environment.yml -y && \
    mamba clean -y --all --force-pkgs-dirs

ENV PATH=/env/bin/:$PATH
ENV CONDA_DEFAULT_ENV=/env
ENV ENV_NAME=/env

ENTRYPOINT ["/_entrypoint.sh"]
CMD ["/bin/bash"]

USER 1000
