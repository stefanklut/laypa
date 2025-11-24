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
RUN mamba env create -f environment.yml && \
    mamba run -n laypa bash -c 'echo CUDA_HOME=$CUDA_HOME; ls -1 $CUDA_HOME/include/cuda_runtime_api.h || echo "cuda_runtime_api.h missing"' && \
    mamba run -n laypa python -c "import torch; print('Torch CUDA available:', torch.cuda.is_available())" && \
    FORCE_CUDA=1 mamba run -n laypa pip install --no-build-isolation ./models/pixel_decoder/ops && \
    mamba clean -y --all --force-pkgs-dirs && \
    rm -r /opt/conda/envs/laypa/nsight* || true

ENV PATH=/opt/conda/envs/laypa/bin:$PATH
ENV CONDA_DEFAULT_ENV=laypa
ENV ENV_NAME=laypa
ENV CUDA_HOME=/opt/conda/envs/laypa
# Architectures: adjust if you only need specific SM versions
ENV TORCH_CUDA_ARCH_LIST=8.0;8.6;9.0
ENV FORCE_CUDA=1

ENTRYPOINT ["/_entrypoint.sh"]
CMD ["/bin/bash"]

USER 1000
