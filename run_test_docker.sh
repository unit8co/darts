#!/bin/bash

docker run -it --entrypoint '/home/jovyan/run_tests.sh' eu.gcr.io/unit8-product/u8timeseries:latest
