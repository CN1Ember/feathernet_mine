#!/usr/bin/env bash
export PYTHONPATH=/home/xiezheng/program2019/insightface_DCP/
export PYTHONPATH=/home/xiezheng/programs2019/insightface_DCP/

python auxnet/main.py auxnet/cifar_resnet.hocon
python channel_selection/main.py channel_selection/cifar10_resnet.hocon
python finetune/main.py finetune/cifar10_resnet.hocon