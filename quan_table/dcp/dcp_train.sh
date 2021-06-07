#!/usr/bin/env bash
export PYTHONPATH=/home/xiezheng/programs2019/insightface_DCP/
python auxnet/main.py auxnet/ms1m_r34_231.hocon
python channel_selection/main.py channel_selection/ms1m_r34_231.hocon
python finetune/main.py finetune/