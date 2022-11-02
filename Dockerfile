FROM conda/miniconda3

RUN apt-get update
RUN apt-get dist-upgrade -y
RUN apt-get clean
RUN apt-get install -y \
    apt-utils git \
    ninja-build gcc g++ \
    ffmpeg libsm6 libxext6

WORKDIR /src/
COPY layout layout-analysis

# When github is open
# RUN git clone https://github.com/stefanklut/layout-analysis.git

WORKDIR /src/layout-analysis
RUN conda update conda
RUN conda install mamba -n base -c conda-forge
RUN conda update --all
RUN conda init bash
RUN conda env create -f environment.yml
RUN conda clean --all

ENV PATH /opt/conda/envs/layout-analysis/bin:$PATH
ENV CONDA_DEFAULT_ENV layout-analysis

SHELL ["conda", "run", "-n", "layout-analysis", "/bin/bash", "-c"]

RUN echo "conda activate layout-analysis" >> ~/.bashrc

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "layout-analysis", "/bin/bash", "-c"]

