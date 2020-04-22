#!/bin/bash
version=$(head -n 1 u8timeseries/VERSION)
docker run -p 8888:8888 -it eu.gcr.io/unit8-product/u8timeseries:${version}
