#!/bin/bash
version=$(head -n 1 u8timeseries/VERSION)
docker run -it --entrypoint '/home/jovyan/work/run_tests.sh' eu.gcr.io/unit8-product/u8timeseries:${version}
