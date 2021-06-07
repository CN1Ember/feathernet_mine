import os

import torch

from insightface_v2.utils.model_analyse import ModelAnalyse
from torch import nn
from torchsummary import summary



class Bottleneck_mobilefacenetv2(nn.Module):
    def __init__(self, in_planes, exp_planes, out_planes, stride):
        super(Bottleneck_mobilefacenetv2, self).__init__()

        self.connect = stride == 1 and in_planes == out_planes
        # planes = in_planes * expansion
        planes = exp_planes

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.prelu1 = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu2 = nn.PReLU(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = self.prelu1(self.bn1(self.conv1(x)))
        out = self.prelu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.connect:
            return x + out
        else:
            return out


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class Mobilefacenetv2_depth_width(nn.Module):

    def __init__(self, embedding_size=512, blocks=[3, 8, 16, 5], width_mult=1.0):
        super(Mobilefacenetv2_depth_width, self).__init__()

        block_setting = [
            # [t, c , n ,s] = [expansion, out_planes, num_blocks, stride]
            [1, 64, blocks[0], 1],
            [2, 64, 1, 2],
            [2, 64, blocks[1], 1],
            [4, 128, 1, 2],
            [2, 128, blocks[2], 1],
            [4, 128, 1, 2],
            [2, 128, blocks[3], 1]
        ]

        self.inplanes = 64
        self.last_planes = 512
        self.last_planes = make_divisible(self.last_planes * width_mult) if width_mult > 1.0 else self.last_planes
        # print("self.last_planes={}".format(self.last_planes))

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.prelu1 = nn.PReLU(self.inplanes)

        self.layers = self._make_layer(Bottleneck_mobilefacenetv2, block_setting, width_mult=width_mult)

        self.conv2 = nn.Conv2d(self.inplanes, self.last_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(self.last_planes)
        self.prelu2 = nn.PReLU(self.last_planes)
        self.conv3 = nn.Conv2d(self.last_planes, self.last_planes, kernel_size=7, groups=self.last_planes,
                               stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(self.last_planes)
        self.linear = nn.Linear(self.last_planes, embedding_size)
        self.bn4 = nn.BatchNorm1d(embedding_size, affine=False)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                # nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, setting, width_mult):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                exp_planes = make_divisible(self.inplanes * t * width_mult)
                out_planes = make_divisible(c * width_mult)
                if i == 0:
                    layers.append(block(self.inplanes, exp_planes, out_planes, s))
                else:
                    layers.append(block(self.inplanes, exp_planes, out_planes, 1))
                self.inplanes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.prelu1(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.prelu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = self.bn4(self.linear(out))
        return out


if __name__ == "__main__":

    # only deepen:model layers_num = 108; model size=8.42 MB; model flops=994.82 M, 994816901.00
    # model = Mobilefacenetv2_depth_width(blocks=[3, 8, 16, 5], embedding_size=512, width_mult=1.0)

    # deepen and widen: model layers_num = 78; model size=8.54 MB; model flops=991.24 M, 991244374.00
    model = Mobilefacenetv2_depth_width(blocks=[2, 6, 10, 4], embedding_size=512, width_mult=1.1)

    print(model)
    summary(model, (3, 112, 112))

    test_input = torch.randn(1, 3, 112, 112)
    model_analyse = ModelAnalyse(model)
    params_num = model_analyse.params_count()
    flops = model_analyse.flops_compute(test_input)
    count = 0
    block_count = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            count = count + 1
        if isinstance(module, Bottleneck_mobilefacenetv2):
            block_count = block_count + 1

    print("\nmodel layers_num = {}".format(count))
    print("model block_count = {}".format(block_count))
    print("model size = {:.2f} MB".format(params_num * 4 / 1024 / 1024))
    print("model flops = {:.2f} ".format(sum(flops)))