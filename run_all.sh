#!/bin/bash

DETECTOR="./build/flow_detector"
DATA_DIR="/home/a/data/TrafficLabelling"

export DETECTOR
find "$DATA_DIR" -type f -name "*.csv" | \
  parallel -j 8 '$DETECTOR "{}"'