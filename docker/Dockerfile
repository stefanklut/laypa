FROM condaforge/mambaforge

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y && \
    apt-get dist-upgrade -y && \
    apt-get clean -y && \
    apt-get install -y \
        apt-utils git \
        ninja-build gcc g++ \
        ffmpeg libsm6 libxext6 && \
    apt-get clean -y && \
    apt-get autoremove -y

WORKDIR /src/
COPY laypa/ laypa/
COPY _entrypoint.sh /_entrypoint.sh

# When github is open
# RUN git clone https://github.com/stefanklut/laypa.git

WORKDIR /src/laypa
# RUN conda update -y --all
# RUN conda init bash
RUN mamba env create -f environment.yml && \
    mamba clean -y --all --force-pkgs-dirs

# SHELL ["conda", "run", "-n", "laypa", "/bin/bash", "-c"]
ENV PATH /opt/conda/envs/laypa/bin:$PATH
ENV CONDA_DEFAULT_ENV laypa
ENV ENV_NAME=laypa

# Fix cuda permissions with the docker
# RUN chmod go+rx /opt/conda/envs/laypa/pkgs/cuda-toolkit/
# Fix git permissions with the docker
RUN git config --system --add safe.directory /src/laypa

ENTRYPOINT ["/_entrypoint.sh"]
CMD ["/bin/bash"]
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "laypa", "/bin/bash", "-c"]

USER 1000
