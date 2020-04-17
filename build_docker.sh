#!/bin/bash

cd "$( dirname "${BASH_SOURCE[0]}" )"
docker build -t eu.gcr.io/unit8-product/u8timeseries:latest .

