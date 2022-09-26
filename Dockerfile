FROM conda/miniconda3

RUN apt-get update && apt-get install -y apt-utils git

RUN git clone https://github.com/stefanklut/layout-analysis.git

RUN cd layout-analysis
RUN conda env create -f environment.yml
