FROM conda/miniconda3

RUN apt-get update
RUN apt-get dist-upgrade -y
RUN apt-get clean
RUN apt-get install -y \
    apt-utils git \
    ninja-build gcc g++ \
    ffmpeg libsm6 libxext6

WORKDIR /src/
COPY laypa laypa

# When github is open
# RUN git clone https://github.com/stefanklut/laypa.git

WORKDIR /src/laypa
RUN conda update conda
RUN conda install mamba -n base -c conda-forge
RUN conda update --all
RUN conda init bash
RUN conda env create -f environment.yml
RUN conda clean --all

ENV PATH /opt/conda/envs/laypa/bin:$PATH
ENV CONDA_DEFAULT_ENV laypa

SHELL ["conda", "run", "-n", "laypa", "/bin/bash", "-c"]

RUN echo "conda activate laypa" >> ~/.bashrc

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "laypa", "/bin/bash", "-c"]

