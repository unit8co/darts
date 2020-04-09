FROM jupyter/base-notebook:python-3.7.6

# By default base-notebook install latest python version and fix it in the pinned constraint
# remove the file, change the python version and update packages

#RUN rm $CONDA_DIR/conda-meta/pinned
#RUN conda install python=3.6.8  -y --quiet && conda update --all -y --quiet && \
#conda clean --all -f -y
RUN conda install -c conda-forge ipywidgets -y --quiet

USER root

RUN apt-get update && apt-get -y install curl && apt-get -y install apt-utils

# to build prophet
RUN apt-get -y install build-essential libc-dev

USER $NB_USER

# u8ts specific deps
RUN pip install pystan
WORKDIR /deps
ADD . /home/jovyan/work
WORKDIR /home/jovyan/work
RUN pip install .

