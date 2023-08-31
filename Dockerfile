FROM ubuntu:latest

# setup packages
RUN apt-get update -y
RUN apt-get install -y python3 python-is-python3 python3-pip default-jre
RUN pip install --upgrade pip

# install python requirements before copying the rest of the files
# this way we can cache the requirements and not have to reinstall them
COPY requirements/ /app/requirements/
RUN pip install -r /app/requirements/dev-all.txt

# copy local files
COPY . /app

# set work directory
WORKDIR /app

# install darts
RUN pip install -e .

# assuming you are working from inside your darts directory:
# docker build . -t darts-test:latest
# docker run -it -v $(pwd)/:/app/ darts-test:latest bash