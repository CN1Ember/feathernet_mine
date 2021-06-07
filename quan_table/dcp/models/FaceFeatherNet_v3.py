# the code base on https://github.com/tonylins/pytorch-mobilenet-v2
import torch.nn as nn
import math
import torch
import sys
sys.path.append("..")
# from torchsummary import summary
from tools.benchmark import compute_speed, stat


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(oup)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(oup)
    )


# reference form : https://github.com/moskomule/senet.pytorch
class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # return x * y
        return x.mul(y)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, downsample=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.downsample = downsample

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(hidden_dim),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            if self.downsample is not None:
                return self.downsample(x) + self.conv(x)
            else:
                return self.conv(x)


class FaceFeatherNet_v3(nn.Module):
    def __init__(self, n_class=2, img_channel=1,input_size=112, se=False, avgdown=False, width_mult=1.0):
        super(FaceFeatherNet_v3, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1024
        self.se = se
        self.avgdown = avgdown
        self.img_channel = img_channel
        self.width_mult = width_mult
        interverted_residual_setting = [

            # t, c, n, s
            # [1, 16, 1, 2],
            # [6, 32, 3, 2],  # 56x56
            # [6, 48, 6, 2],  # 14x14
            # [6, 64, 4, 2],  # 7x7

            # # t, c, n, s
            # [1, 16, 1, 1],
            # [6, 32, 3, 2],  # 56x56
            # [6, 48, 6, 2],  # 14x14
            # [6, 64, 4, 2],  # 7x7

            # t, c, n, s
            [1, 16, 1, 2],
            [6, 32, 2, 2],  # 56x56
            [6, 48, 6, 2],  # 14x14
            [6, 64, 3, 2],  # 7x7

            # # t, c, n, s deeper
            # [1, 16, 1, 2],
            # [6, 32, 4, 2],  # 56x56
            # [6, 48, 6, 2],  # 14x14
            # [6, 64, 3, 2],  # 7x7

        ]

        # building first layer
        assert input_size % 16 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(img_channel, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                downsample = None
                if i == 0:
                    if self.avgdown and s != 1:
                        downsample = nn.Sequential(nn.AvgPool2d(2, stride=2),
                                                   nn.BatchNorm2d(input_channel),
                                                   nn.Conv2d(input_channel, output_channel, kernel_size=1, bias=False))
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t, downsample=downsample))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t, downsample=downsample))
                input_channel = output_channel
            if self.se:
                self.features.append(SELayer(input_channel))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        #         building last several layers
        self.final_DW = nn.Sequential(nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=2, padding=1,
                                                groups=input_channel, bias=False))

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.final_DW(x)

        x = x.view(x.size(0), -1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def FaceFeatherNetA_v3(se=False, width_mult=1.0):
    model = FaceFeatherNet_v3(se = se, width_mult=width_mult)
    return model

def FaceFeatherNetB_v3(se=True, img_channel=1, width_mult=1.0):
    model = FaceFeatherNet_v3(se=se, avgdown=True, img_channel=img_channel, width_mult=width_mult)
    return model


# if __name__ == "__main__":
#     # model = FaceFeatherNetB_v2()         # Total Flops(Conv Only): 70.46MFlops, model size = 1.36MB
#     model = FaceFeatherNetA_v3(se=False)  # Total Flops(Conv Only): 70.46MFlops, model size = 1.35MB
#     print(model)
#
#     str_input_size = '1x1x112x112'
#     input_size = tuple(int(x) for x in str_input_size.split('x'))
#     stat(model, input_size)

    # summary(model, (1, 112, 112))


