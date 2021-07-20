FROM jupyter/base-notebook:python-3.9.5

RUN conda update --all -y --quiet \
 && conda install -c conda-forge ipywidgets -y --quiet \
 && conda clean --all -f -y

USER root

RUN apt-get update && apt-get -y install curl && apt-get -y install apt-utils

# to build prophet
RUN apt-get -y install build-essential libc-dev

USER $NB_USER

# u8ts specific deps
RUN pip install pystan
ADD . /home/jovyan/work

WORKDIR /home/jovyan/work

RUN pip install .['all']
