#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PYTHONPATH:$DIR/../../
export PYTHONPATH=$PYTHONPATH:$DIR/../../slim

python3 $DIR/../train.py \
--train_dir $DIR/../../object_detection/data/ssd_mobilenet_v1_coco_11_06_2017 \
--pipeline_config_path $DIR/../../object_detection/samples/configs/ssd_mobilenet_beer.config

