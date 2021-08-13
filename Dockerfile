FROM jupyter/base-notebook:python-3.9.5

RUN conda update --all -y --quiet \
 && conda install -c conda-forge ipywidgets -y --quiet \
 && conda clean --all -f -y

USER root

# to build pystan
RUN apt-get update \
 && apt-get -y install build-essential \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

USER $NB_USER

ADD . /home/jovyan/work

WORKDIR /home/jovyan/work

RUN pip install .
