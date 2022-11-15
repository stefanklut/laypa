FROM continuumio/miniconda3

RUN apt-get update
RUN apt-get dist-upgrade -y
RUN apt-get clean
RUN apt-get install -y \
    apt-utils git \
    ninja-build gcc g++ \
    ffmpeg libsm6 libxext6

WORKDIR /src/
COPY laypa/ laypa/
COPY _entrypoint.sh /usr/local/bin/_entrypoint.sh

# When github is open
# RUN git clone https://github.com/stefanklut/laypa.git

WORKDIR /src/laypa
RUN conda update -y --all
RUN conda init bash
RUN conda env create -f environment.yml
RUN conda clean --all

# RUN conda install -y mamba -n base -c conda-forge
# RUN mamba update -y --all
# RUN mamba init bash
# RUN mamba env update -n base -f environment.yml
# RUN mamba clean --all

SHELL ["conda", "run", "-n", "laypa", "/bin/bash", "-c"]
RUN echo "conda activate laypa" >> ~/.bashrc
ENV PATH /opt/conda/envs/laypa/bin:$PATH
ENV CONDA_DEFAULT_ENV laypa
ENV ENV_NAME=laypa

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["/bin/bash"]
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "laypa", "/bin/bash", "-c"]