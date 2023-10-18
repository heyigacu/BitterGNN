FROM continuumio/miniconda3


WORKDIR /app


ADD . /app


RUN conda env create -f environment.yml

RUN echo "source activate bittergnn" > ~/.bashrc
ENV PATH /opt/conda/envs/bittergnn/bin:$PATH
RUN pip install scikit-learn torch
