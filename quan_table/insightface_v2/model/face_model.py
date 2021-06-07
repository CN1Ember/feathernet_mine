#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/14 17:23
# @Author  : xiezheng
# @Site    : 
# @File    : face_model.py

from torch import nn

from insightface_v2.model.insightface_resnet import LResNet34E_IR

from insightface_v2.model.models import MobileFaceNet, ArcMarginModel
from insightface_v2.model.mobilenetv3 import MobileNetV3_Large
from insightface_v2.model.zq_mobilefacenet_old import ZQMobilefacenet



class face_model(nn.Module):
    def __init__(self, args, num_classes):
        super(face_model, self).__init__()

        if args.network == 'LResNet34E_IR':
            self.model = LResNet34E_IR()
        elif args.network == 'mobilefacenet_v1':
            self.model = MobileFaceNet(embedding_size=args.emb_size, blocks=args.block_setting)  # mobilefacenet_v1
        elif args.network == 'mobilefacenet_v2':
            self.model = MobileFaceNet(embedding_size=args.emb_size, blocks=args.block_setting)  # mobilefacenet_v2
        elif args.network == 'zq_mobilefacenet':
            self.model = ZQMobilefacenet(embedding_size=args.emb_size, blocks=args.block_setting)
        elif args.network == 'mobilenet_v3':
            self.model = MobileNetV3_Large(embedding_size=args.emb_size)

        self. metric_fc = ArcMarginModel(args, num_classes, args.emb_size)

    def forward(self, x, label):
        out = self.model(x)
        out = self.metric_fc(out, label)
        return out
