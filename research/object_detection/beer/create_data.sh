#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PYTHONPATH:$DIR/../../
export PYTHONPATH=$PYTHONPATH:$DIR/../../slim
DATA=/home/admins/data/beer_data
echo "generating train dataset"
python3 $DIR/../../object_detection/dataset_tools/create_beer_tf_record.py \
--data_dir $DATA \
--set train \
--output_path $DATA/train.record \
--label_map_path $DIR/../../object_detection/data/beer_label_map.pbtxt
echo "generating val dataset"
python3 $DIR/../../object_detection/dataset_tools/create_beer_tf_record.py \
--data_dir $DATA \
--set val \
--output_path $DATA/val.record \
--label_map_path $DIR/../../object_detection/data/beer_label_map.pbtxt