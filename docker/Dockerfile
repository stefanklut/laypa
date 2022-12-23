FROM condaforge/mambaforge

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y && \
    apt-get dist-upgrade -y && \
    apt-get clean -y && \
    apt-get install -y \
        apt-utils git \
        ninja-build gcc g++ \
        ffmpeg libsm6 libxext6 && \
        apt-get autoremove && \
    apt-get clean -y

WORKDIR /src/
COPY laypa/ laypa/
COPY _entrypoint.sh /usr/local/bin/_entrypoint.sh

# When github is open
# RUN git clone https://github.com/stefanklut/laypa.git

WORKDIR /src/laypa
# RUN conda update -y --all
# RUN conda init bash
RUN --mount=type=cache,target=/opt/conda/pkgs mamba env create -f environment.yml && \
    mamba clean -y --all

SHELL ["conda", "run", "-n", "laypa", "/bin/bash", "-c"]
RUN echo "conda activate laypa" >> ~/.bashrc
ENV PATH /opt/conda/envs/laypa/bin:$PATH
ENV CONDA_DEFAULT_ENV laypa
ENV ENV_NAME=laypa

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["/bin/bash"]
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "laypa", "/bin/bash", "-c"]

USER 1000