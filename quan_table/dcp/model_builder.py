import dcp.models as md


def get_model(setting, dataset, net_type, depth, n_classes, pruning_rate=0.0):
    """
    Available model
    cifar:
        preresnet
        vgg
    imagenet:
        resnet
    """

    if dataset in ["cifar10", "cifar100"]:
        if net_type == "preresnet":
            model = md.PreResNet(depth=depth, num_classes=n_classes)
        elif net_type == "vgg":
            model = md.VGG_CIFAR(depth=depth, num_classes=n_classes)
        else:
            assert False, "use {} data while network is {}".format(dataset, net_type)

    elif dataset in ["imagenet", "imagenet_mio"]:
        if net_type == "resnet":
            model = md.ResNet(depth=depth, num_classes=n_classes)
        else:
            assert False, "use {} data while network is {}".format(dataset, net_type)

    elif dataset in ["ms1m_v2", 'sub_iccv_ms1m', 'sub_webface_0.1']:
        if net_type == "LResnetxE-IR":
            if depth == 18:
                model = md.LResNet18E_IR()
            elif depth == 34:
                model = md.LResNet34E_IR()

        elif net_type == "mobilefacenet_v1":
            model = md.Mobilefacenet(embed_size=setting.embed_size)
        elif net_type == "mobilefacenet_v1_p0.5":
            # model = md.pruned_Mobilefacenet(pruning_rate=0.5)   # old
            model = md.Mobilefacenetv2_width_wm(embedding_size=setting.embed_size, pruning_rate=0.5) # new

        else:
            assert False, "unsupported net_type: {}".format(net_type)

    elif dataset in ['iccv_ms1m', 'sub_iccv_ms1m', 'sub_webface_0.1']:
        if net_type == "zq_mobilefacenet":
            model = md.ZQMobilefacenet(blocks=[4,8,16,4], embedding_size=512, fc_type=setting.fc_type)
        elif net_type == "mobilefacenet_v2":
            model = md.Mobilefacenetv2(embedding_size=512, blocks=[3,8,16,5], fc_type=setting.fc_type)
        elif net_type == "mobilefacenet_v1":
            model = md.Mobilefacenet(embed_size=setting.embed_size)

        elif net_type == "mobilefacenet_v1_p0.5":
            # model = md.pruned_Mobilefacenet(pruning_rate=0.5)   # old
            model = md.Mobilefacenetv2_width_wm(embedding_size=setting.embed_size, pruning_rate=0.5) # new

    else:
        assert False, "unsupported data set: {}".format(dataset)

    return model
