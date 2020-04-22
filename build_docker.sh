#!/bin/bash
cd "$( dirname "${BASH_SOURCE[0]}" )"
version=$(head -n 1 u8timeseries/VERSION)
docker build -t eu.gcr.io/unit8-product/u8timeseries:${version} -t eu.gcr.io/unit8-product/u8timeseries:latest .

