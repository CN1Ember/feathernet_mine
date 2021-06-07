import logging

import torch.nn as nn

from dcp.mask_conv import MaskConv2d, MaskLinear

__all__ = ['ResModelPrune', 'get_select_channels']

logger = logging.getLogger('channel_selection')


def get_select_channels(d):
    """
    Get select channels
    """

    select_channels = (d > 0).nonzero().squeeze()
    return select_channels


def get_thin_params(layer, select_channels, dim=0):
    """
    Get params from layers after pruning
    """

    if isinstance(layer, (nn.Conv2d, MaskConv2d)):
        if isinstance(layer, MaskConv2d):
            layer.weight.data = layer.pruned_weight.clone().data
        thin_weight = layer.weight.data.index_select(dim, select_channels)
        if layer.bias is not None:
            if dim == 0:
                thin_bias = layer.bias.data.index_select(dim, select_channels)
            else:
                thin_bias = layer.bias.data
        else:
            thin_bias = None

    elif isinstance(layer, (nn.Linear, MaskLinear)):
        if isinstance(layer, MaskLinear):
            layer.weight.data = layer.pruned_weight.clone().data
        thin_weight = layer.weight.data.index_select(dim, select_channels)
        if layer.bias is not None:
            if dim == 0:
                thin_bias = layer.bias.data.index_select(dim, select_channels)
            else:
                thin_bias = layer.bias.data
        else:
            thin_bias = None

    elif isinstance(layer, nn.BatchNorm2d):
        assert dim == 0, "invalid dimension for bn_layer"

        thin_weight = layer.weight.data.index_select(dim, select_channels)
        thin_mean = layer.running_mean.index_select(dim, select_channels)
        thin_var = layer.running_var.index_select(dim, select_channels)
        if layer.bias is not None:
            thin_bias = layer.bias.data.index_select(dim, select_channels)
        else:
            thin_bias = None
        return (thin_weight, thin_mean), (thin_bias, thin_var)

    elif isinstance(layer, nn.PReLU):
        thin_weight = layer.weight.data.index_select(dim, select_channels)
        thin_bias = None

    return thin_weight, thin_bias


def replace_layer(old_layer, init_weight, init_bias=None, keeping=False):
    """
    Replace specific layer of model
    :params layer: original layer
    :params init_weight: thin_weight
    :params init_bias: thin_bias
    :params keeping: whether to keep MaskConv2d
    """

    if hasattr(old_layer, "bias") and old_layer.bias is not None:
        bias_flag = True
    else:
        bias_flag = False
    if isinstance(old_layer, MaskConv2d) and keeping:
        new_layer = MaskConv2d(
            init_weight.size(1),
            init_weight.size(0),
            kernel_size=old_layer.kernel_size,
            stride=old_layer.stride,
            padding=old_layer.padding,
            bias=bias_flag)

        new_layer.pruned_weight.data.copy_(init_weight)
        new_layer.weight.data.copy_(init_weight)
        if init_bias is not None:
            new_layer.bias.data.copy_(init_bias)
        new_layer.d.copy_(old_layer.d)

    elif isinstance(old_layer, (nn.Conv2d, MaskConv2d)):
        if old_layer.groups != 1:
            new_groups = init_weight.size(0)
            in_channels = init_weight.size(0)
            out_channels = init_weight.size(0)
        else:
            new_groups = 1
            in_channels = init_weight.size(1)
            out_channels = init_weight.size(0)

        new_layer = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=old_layer.kernel_size,
                              stride=old_layer.stride,
                              padding=old_layer.padding,
                              bias=bias_flag,
                              groups=new_groups)

        new_layer.weight.data.copy_(init_weight)
        if init_bias is not None:
            new_layer.bias.data.copy_(init_bias)

    elif isinstance(old_layer, (nn.Linear, MaskLinear)):
        in_channels = init_weight.size(1)
        out_channels = init_weight.size(0)
        new_layer = nn.Linear(in_channels, out_channels, bias=bias_flag,)
        new_layer.weight.data.copy_(init_weight)
        if init_bias is not None:
            new_layer.bias.data.copy_(init_bias)

    elif isinstance(old_layer, nn.BatchNorm2d):
        weight = init_weight[0]
        mean_ = init_weight[1]
        bias = init_bias[0]
        var_ = init_bias[1]
        new_layer = nn.BatchNorm2d(weight.size(0))
        new_layer.weight.data.copy_(weight)
        assert init_bias is not None, "batch normalization needs bias"
        new_layer.bias.data.copy_(bias)
        new_layer.running_mean.copy_(mean_)
        new_layer.running_var.copy_(var_)

    elif isinstance(old_layer, nn.PReLU):
        print(init_weight.size(0))
        new_layer = nn.PReLU(init_weight.size(0))
        # print("init_weight shape=", init_weight.size())
        new_layer.weight.data.copy_(init_weight)

    else:
        assert False, "unsupport layer type:" + \
                      str(type(old_layer))
    return new_layer


class SeqModelPrune(object):
    def __init__(self, model, net_type):
        self.model = model
        self.net_type = net_type
        logger.info("|===>init ModelPrune")

    def run(self):
        # divide model into several segment
        if self.net_type in ["vgg"]:
            feature_prune = SeqPrune(self.model.features)
            feature_prune.pruning()
            self.model.features = feature_prune.segment
            self.model.cuda()
            logger.info(self.model)

        else:
            assert False, "invalid net_type: " + self.net_type


# -----------------------------------------------------------------------------------------------------------------

class SeqPrune(object):
    def __init__(self, segment):
        """
        Only support convolutional layer, but not fc layer
        :params segment: target segment or sequential
        """

        self.segment = segment
        self.segment_length = None
        self.select_channels = None
        logger.info("|===>init SegmentPrune")

        # get length of segment
        if isinstance(self.segment, nn.DataParallel):
            self.segment = list(self.segment.module)
        elif isinstance(self.segment, nn.Sequential):
            self.segment = list(self.segment)
        else:
            self.segment = [self.segment]
        self.segment_length = len(self.segment)
        # self.segment = nn.Sequential(*self.segment)
        # self.segment.cuda()

    def pruning(self):
        logger.info("|===>pruning layers")
        for i in range(self.segment_length):
            if isinstance(self.segment[i], MaskConv2d):

                # compute selected channels
                select_channels = get_select_channels(self.segment[i].d)
                # print "check channels:", self.segment[i].d.sum()

                # replace current layer
                thin_weight, thin_bias = get_thin_params(
                    self.segment[i], select_channels, 1)

                self.segment[i] = replace_layer(
                    self.segment[i], thin_weight, thin_bias)
                # print "check layer:", self.segment[i]

                self.segment[i].cuda()
                for j in range(i - 1, -1, -1):
                    if isinstance(self.segment[j], (nn.Conv2d, nn.BatchNorm2d, MaskConv2d)):
                        thin_weight, thin_bias = get_thin_params(
                            self.segment[j], select_channels, 0)
                        self.segment[j] = replace_layer(
                            self.segment[j], thin_weight, thin_bias)
                        self.segment[j].cuda()
                        if isinstance(self.segment[j], nn.Conv2d) and self.segment[j].groups == 1:
                            break

        self.segment = nn.Sequential(*self.segment)
        self.segment.cuda()
        # assert False
        return select_channels


# resnet only ---------------------------------------------------------------------------
class ResBlockPrune(object):
    """
    Residual block pruning
    """

    def __init__(self, block, block_type):
        self.block = block
        self.block_type = block_type
        self.select_channels = None

    def pruning(self):
        """
        Perform pruning
        """

        # prune pre-resnet on cifar
        if self.block_type in ["preresnet"]:
            if self.block.conv2.d.sum() == 0:
                self.block = self.block.downsample
                logger.info("remove whole block")
                return None
            # compute selected channels
            select_channels = get_select_channels(self.block.conv2.d)
            self.select_channels = select_channels

            # prune and replace conv2
            thin_weight, thin_bias = get_thin_params(self.block.conv2, select_channels, 1)
            self.block.conv2 = replace_layer(self.block.conv2, thin_weight, thin_bias)

            # prune and replace bn2
            thin_weight, thin_bias = get_thin_params(self.block.bn2, select_channels, 0)
            self.block.bn2 = replace_layer(self.block.bn2, thin_weight, thin_bias)

            # prune and replace conv1
            thin_weight, thin_bias = get_thin_params(self.block.conv1, select_channels, 0)
            self.block.conv1 = replace_layer(self.block.conv1, thin_weight, thin_bias)
            # self.block.conv1 = replace_layer(self.block.conv1, thin_weight, thin_bias, keeping=True)
            self.block.cuda()

        # prune shallow resnet on imagenet
        elif self.block_type == "resnet_irblock":
            if self.block.conv2.d.sum() == 0:
                self.block = self.block.downsample
                logger.info("remove whole block")
                return None

            # compute selected channels
            select_channels = get_select_channels(self.block.conv2.d)
            self.select_channels = select_channels

            # prune and replace conv2
            thin_weight, thin_bias = get_thin_params(self.block.conv2, select_channels, 1)
            self.block.conv2 = replace_layer(self.block.conv2, thin_weight, thin_bias)

            # prune and replace prelu
            # thin_weight, thin_bias = get_thin_params(self.block.prelu, select_channels, 0)
            # self.block.prelu = replace_layer(self.block.prelu, thin_weight, thin_bias)

            # prune and replace bn1
            thin_weight, thin_bias = get_thin_params(self.block.bn1, select_channels, 0)
            self.block.bn1 = replace_layer(self.block.bn1, thin_weight, thin_bias)

            # prune and replace conv1
            thin_weight, thin_bias = get_thin_params(self.block.conv1, select_channels, 0)
            self.block.conv1 = replace_layer(self.block.conv1, thin_weight, thin_bias)

        # prune shallow resnet on imagenet
        elif self.block_type == "resnet_basic":
            if self.block.conv2.d.sum() == 0:
                self.block = self.block.downsample
                logger.info("remove whole block")
                return None

            # compute selected channels
            select_channels = get_select_channels(self.block.conv2.d)
            self.select_channels = select_channels

            # prune and replace conv2
            thin_weight, thin_bias = get_thin_params(self.block.conv2, select_channels, 1)
            self.block.conv2 = replace_layer(self.block.conv2, thin_weight, thin_bias)

            # prune and replace bn1
            thin_weight, thin_bias = get_thin_params(self.block.bn1, select_channels, 0)
            self.block.bn1 = replace_layer(self.block.bn1, thin_weight, thin_bias)

            # prune and replace conv1
            thin_weight, thin_bias = get_thin_params(self.block.conv1, select_channels, 0)
            self.block.conv1 = replace_layer(self.block.conv1, thin_weight, thin_bias)

        # prune deep resnet on imagenet
        elif self.block_type == "resnet_bottleneck":
            if (self.block.conv2.d.sum() == 0
                    or self.block.conv3.d.sum() == 0):
                self.block = self.block.downsample
                logger.info("remove whole block")
                return None

            # compute selected channels of conv2
            select_channels = get_select_channels(self.block.conv2.d)
            self.select_channels = select_channels

            # prune and replace conv2
            thin_weight, thin_bias = get_thin_params(self.block.conv2, select_channels, 1)
            self.block.conv2 = replace_layer(self.block.conv2, thin_weight, thin_bias)

            # prune and replace bn1
            thin_weight, thin_bias = get_thin_params(self.block.bn1, select_channels, 0)
            self.block.bn1 = replace_layer(self.block.bn1, thin_weight, thin_bias)

            # prune and replace conv1
            thin_weight, thin_bias = get_thin_params(self.block.conv1, select_channels, 0)
            self.block.conv1 = replace_layer(self.block.conv1, thin_weight, thin_bias)  # , keeping=True)

            self.block.cuda()
            # compute selected channels of conv3
            select_channels = get_select_channels(self.block.conv3.d)

            # prune and replace conv3
            thin_weight, thin_bias = get_thin_params(self.block.conv3, select_channels, 1)
            self.block.conv3 = replace_layer(self.block.conv3, thin_weight, thin_bias)

            # prune and replace bn2
            thin_weight, thin_bias = get_thin_params(self.block.bn2, select_channels, 0)
            self.block.bn2 = replace_layer(self.block.bn2, thin_weight, thin_bias)

            # prune and replace conv2
            thin_weight, thin_bias = get_thin_params(self.block.conv2, select_channels, 0)
            self.block.conv2 = replace_layer(self.block.conv2, thin_weight, thin_bias)
            self.block.cuda()

        elif self.block_type == "mobilefacenet_v1":
            if self.block.conv3.d.sum() == 0:
                self.block = self.block.shortcut
                logger.info("remove whole block")
                return None

            # compute selected channels of conv3
            select_channels = get_select_channels(self.block.conv3.d)
            self.select_channels = select_channels

            # prune and replace conv3
            thin_weight, thin_bias = get_thin_params(
                self.block.conv3, select_channels, 1)
            self.block.conv3 = replace_layer(
                self.block.conv3, thin_weight, thin_bias)

            # prune and replace prelu
            thin_weight, thin_bias = get_thin_params(self.block.prelu2, select_channels, 0)
            self.block.prelu2 = replace_layer(self.block.prelu2, thin_weight, thin_bias)

            # prune and replace bn2
            thin_weight, thin_bias = get_thin_params(
                self.block.bn2, select_channels, 0)
            self.block.bn2 = replace_layer(
                self.block.bn2, thin_weight, thin_bias)

            # prune and replace conv2
            thin_weight, thin_bias = get_thin_params(
                self.block.conv2, select_channels, 0)
            self.block.conv2 = replace_layer(
                self.block.conv2, thin_weight, thin_bias)

            # prune and replace prelu
            thin_weight, thin_bias = get_thin_params(self.block.prelu1, select_channels, 0)
            self.block.prelu1 = replace_layer(self.block.prelu1, thin_weight, thin_bias)

            # prune and replace bn1
            thin_weight, thin_bias = get_thin_params(
                self.block.bn1, select_channels, 0)
            self.block.bn1 = replace_layer(
                self.block.bn1, thin_weight, thin_bias)

            # (1) pruned conv1: conv1 --> mask_conv
            # prune and replace conv1
            # thin_weight, thin_bias = get_thin_params(
            #     self.block.conv1, select_channels, 0)
            # self.block.conv1 = replace_layer(
            #     self.block.conv1, thin_weight, thin_bias, keeping=True)

            # (2) don't pruned conv1
            thin_weight, thin_bias = get_thin_params(
                self.block.conv1, select_channels, 0)
            self.block.conv1 = replace_layer(
                self.block.conv1, thin_weight, thin_bias)

            # self.block.cuda()

        else:
            assert False, "invalid block type: " + self.block_type


class ResSeqPrune(object):
    """
    Sequantial pruning
    """

    def __init__(self, sequential, seq_type):
        self.sequential = sequential
        self.sequential_length = len(list(self.sequential))
        self.res_block_prune = []
        self.select_channels = None

        for i in range(self.sequential_length):
            self.res_block_prune.append(ResBlockPrune(self.sequential[i], block_type=seq_type))

    def pruning(self):
        """
        Perform pruning
        """

        for i in range(self.sequential_length):
            self.res_block_prune[i].pruning()

        temp_seq = []
        for i in range(self.sequential_length):
            if self.res_block_prune[i].block is not None:
                temp_seq.append(self.res_block_prune[i].block)
        self.sequential = nn.Sequential(*temp_seq)
        self.select_channels = self.res_block_prune[0].select_channels


class ResModelPrune(object):
    """
    Prune residual networks
    """

    def __init__(self, model, net_type, depth):
        self.model = model
        if net_type == "resnet":
            if depth >= 50:
                self.net_type = "resnet_bottleneck"
            else:
                self.net_type = "resnet_basic"

        elif net_type == "LResnetxE-IR":
            self.net_type = "resnet_irblock"

        else:
            self.net_type = net_type
        logger.info("|===>Init ResModelPrune")
        self.select_channels = None

    def run(self):
        """
        Perform pruning
        """

        if self.net_type in ["resnet_basic", "resnet_bottleneck", "resnet_irblock"]:
            res_seq_prune = [
                ResSeqPrune(self.model.layer1, seq_type=self.net_type),
                ResSeqPrune(self.model.layer2, seq_type=self.net_type),
                ResSeqPrune(self.model.layer3, seq_type=self.net_type),
                ResSeqPrune(self.model.layer4, seq_type=self.net_type)
            ]

            for i in range(4):
                res_seq_prune[i].pruning()
            self.model.layer1 = res_seq_prune[0].sequential
            self.model.layer2 = res_seq_prune[1].sequential
            self.model.layer3 = res_seq_prune[2].sequential
            self.model.layer4 = res_seq_prune[3].sequential
            self.model.cuda()
            logger.info(self.model)

        elif self.net_type in ["preresnet"]:
            res_seq_prune = [
                ResSeqPrune(self.model.layer1, seq_type=self.net_type),
                ResSeqPrune(self.model.layer2, seq_type=self.net_type),
                ResSeqPrune(self.model.layer3, seq_type=self.net_type)
            ]
            for i in range(3):
                res_seq_prune[i].pruning()

            self.model.layer1 = res_seq_prune[0].sequential
            self.model.layer2 = res_seq_prune[1].sequential
            self.model.layer3 = res_seq_prune[2].sequential
            logger.info(self.model)
            self.model.cuda()

        elif self.net_type == "mobilefacenet_v1":
            res_seq_prune = ResSeqPrune(self.model.layers, seq_type=self.net_type)
            res_seq_prune.pruning()
            self.model.layers = res_seq_prune.sequential

            if self.net_type == "mobilefacenet_v1" and isinstance(self.model.linear, MaskLinear):

                # compute selected channels of conv3
                select_channels = get_select_channels(self.model.linear.d)
                self.select_channels = select_channels

                # prune and replace linear
                thin_weight, thin_bias = get_thin_params(self.model.linear, select_channels, 1)
                self.model.linear = replace_layer(self.model.linear, thin_weight, thin_bias)

                # prune and replace bn4
                thin_weight, thin_bias = get_thin_params(self.model.bn4, select_channels, 0)
                self.model.bn4 = replace_layer(self.model.bn4, thin_weight, thin_bias)

                # prune and replace conv4
                thin_weight, thin_bias = get_thin_params(self.model.conv4, select_channels, 0)
                self.model.conv4 = replace_layer(self.model.conv4, thin_weight, thin_bias)

                # prune and replace prelu3
                thin_weight, thin_bias = get_thin_params(self.model.prelu3, select_channels, 0)
                self.model.prelu3 = replace_layer(self.model.prelu3, thin_weight, thin_bias)

                # prune and replace bn3
                thin_weight, thin_bias = get_thin_params(self.model.bn3, select_channels, 0)
                self.model.bn3 = replace_layer(self.model.bn3, thin_weight, thin_bias)

                # prune and replace conv3
                thin_weight, thin_bias = get_thin_params(self.model.conv3, select_channels, 0)
                self.model.conv3 = replace_layer(self.model.conv3, thin_weight, thin_bias)

            logger.info(self.model)
            self.model.cuda()
