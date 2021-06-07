#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/7 10:49
# @Author  : xiezheng
# @Site    : 
# @File    : hard_link.py

import os, sys

# 打开文件
path = "./foo.txt"
fd = os.open(path, os.O_RDWR | os.O_CREAT)

# 关闭文件
os.close(fd)

# 创建以上文件的拷贝
dst = "./hard.txt"
os.link(path, dst)

print("创建硬链接成功!!")
