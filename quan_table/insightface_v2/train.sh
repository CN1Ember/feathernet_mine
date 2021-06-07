#!/usr/bin/env bash
# prepare data
export PYTHONPATH=/home/xiezheng/program2019/insightface_DCP/
python insightface_data/prepare_data.py

# train
python train.py