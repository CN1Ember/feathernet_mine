#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/3 13:13
# @Author  : xiezheng
# @Site    : 
# @File    : process_txt.py

import os

txt_path = "./log.txt"


for line in open(txt_path, 'r'):
    line = line.strip()
