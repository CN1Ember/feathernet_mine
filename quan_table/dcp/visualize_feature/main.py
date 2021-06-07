import os

import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt

from dcp.mask_conv import MaskConv2d
from dcp.models.resnet import ResNet, BasicBlock, Bottleneck

depth = 18
n_classes = 1000
conv_count = 0
relu_count = 0
block_count = 0


def replace_layer_with_mask_conv_resnet(model):
    """
    Replace the conv layer in resnet with mask_conv for ResNet
    """

    for module in model.modules():
        if isinstance(module, (BasicBlock)):
            # replace conv2
            temp_conv = MaskConv2d(
                in_channels=module.conv2.in_channels,
                out_channels=module.conv2.out_channels,
                kernel_size=module.conv2.kernel_size,
                stride=module.conv2.stride,
                padding=module.conv2.padding,
                bias=(module.conv2.bias is not None))

            temp_conv.weight.data.copy_(module.conv2.weight.data)
            if module.conv2.bias is not None:
                temp_conv.bias.data.copy_(module.conv2.bias.data)
            module.conv2 = temp_conv

            if isinstance(module, Bottleneck):
                # replace conv3
                temp_conv = MaskConv2d(
                    in_channels=module.conv3.in_channels,
                    out_channels=module.conv3.out_channels,
                    kernel_size=module.conv3.kernel_size,
                    stride=module.conv3.stride,
                    padding=module.conv3.padding,
                    bias=(module.conv3.bias is not None))

                temp_conv.weight.data.copy_(module.conv3.weight.data)
                if module.conv3.bias is not None:
                    temp_conv.bias.data.copy_(module.conv3.bias.data)
                module.conv3 = temp_conv
    return model


def set_cpu():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)


def draw_fig(vis_input, mask, prefix, tag, step):
    # output_img = torchvision.utils.make_grid(vis_output.squeeze(0).unsqueeze(1),
    #                                          nrow=int(
    #                                              math.sqrt(vis_output.size(1))),
    #                                          normalize=True)
    vis_input = vis_input
    np_img = vis_input.cpu().numpy()
    mask = mask.cpu().numpy()
    np_img = np_img[0]
    n = np_img.shape[0]
    # print(np_img.shape)
    # assert False
    # np_img = np.reshape(np_img, (1, np_img.shape[0], np_img.shape[1]))
    # self.logger.image_summary(tag, np_img, step)

    # write to file
    for i in range(n):
        plt.figure()
        plt.imshow(np_img[i], cmap='jet')
        # plt.imshow(np_img[i])
        plt.axis('off')
        if mask[i] == 0:
            plt.savefig(prefix + '/pruned/resnet18-featuremap-pruned-' + tag + '-' + str(i) + '.png')
        else:
            plt.savefig(prefix + '/selected/resnet18-featuremap-selected-' + tag + '-' + str(i) + '.png')
    # plt.savefig('./fig/resnet18-featuremap-' + tag + '.pdf', format='pdf')
    # plt.close()


def block_hook(module, input, output):
    block_count = module.block_count
    prefix = "H:/TPAMI_channel_pruning/visualization/fig/{}".format(block_count)

    if not os.path.isdir(prefix):
        os.mkdir(prefix)

    if not os.path.isdir(os.path.join(prefix, 'pruned')):
        os.mkdir(os.path.join(prefix, 'pruned'))

    if not os.path.isdir(os.path.join(prefix, 'selected')):
        os.mkdir(os.path.join(prefix, 'selected'))

    if isinstance(module, (MaskConv2d, BasicBlock, Bottleneck)):
        tag = "block" + str(block_count)
        step = block_count
        vis_input = input[0].data
    else:
        assert False, "not a residual block but register a block_hook %s" + \
                      str(type(module))

    mask = module.d.reshape(-1)
    nonzero_mask = module.d.nonzero().reshape(-1)
    num_select = int(module.d.sum().item())
    if vis_input is not None:
        draw_fig(vis_input, mask, prefix, tag, step)
    # select_input = vis_input.index_select(1, nonzero_mask)
    # # print(select_input.shape)
    # select_input = select_input.reshape(num_select, -1)
    # select_input_norm = torch.norm(select_input, 2, 1)
    # select_input = select_input / select_input_norm.reshape(-1, 1)
    #
    # cosine_matrix = torch.matmul(select_input, torch.transpose(select_input, 0, 1))
    # cosine_matrix = cosine_matrix.cpu().numpy()
    #
    # triu_index = np.triu_indices(num_select, 1)
    # cosine_vector = cosine_matrix[triu_index]
    # plt.hist(cosine_vector, bins='auto')
    # plt.show()
    # print(num_select)
    # print(cosine_vector)
    # print(cosine_vector.shape)
    # print(triu_index)
    # cosine_matrix = np.triu(cosine_matrix)
    # print(cosine_matrix)


    # plt.figure()
    # # plt.imshow(cosine_matrix, cmap='jet')
    # plt.imshow(cosine_matrix)
    # plt.colorbar()
    # # plt.imshow(np_img[i])
    # plt.axis('off')
    # plt.savefig(prefix + '-score.png')
    # print(cosine_matrix.shape)
    # print(select_input.shape)


def single_input(img_path):
    test_img = Image.open(img_path).convert('RGB')
    pre_processing = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return pre_processing(test_img)


def run(save_path, img_path, model):
    if not os.path.isdir(os.path.join(save_path, "fig")):
        os.mkdir(os.path.join(save_path, "fig"))

    # fhook = self.model.register_forward_hook(self.hook)
    block_count = 1
    model.layer1[0].conv2.block_count = block_count
    model.layer1[0].conv2.register_forward_hook(block_hook)
    # for layer in model.modules():
    #     # if isinstance(layer, nn.Conv2d):
    #     #     layer.register_forward_hook(self.conv_hook)
    #     # elif isinstance(layer, nn.ReLU):
    #     #     layer.register_forward_hook(self.relu_hook)
    #     if isinstance(layer, (BasicBlock, Bottleneck)):
    #         # print('test')
    #         block_count += 1
    #
    #         # layer.block_count = block_count
    #         # layer.register_forward_hook(block_hook)
    #         if hasattr(layer, 'conv3'):
    #             layer.conv3.block_count = block_count
    #             layer.conv3.register_forward_hook(block_hook)
    #         elif hasattr(layer, 'conv2'):
    #             layer.conv2.block_count = block_count
    #             layer.conv2.register_forward_hook(block_hook)

    # get sample from train_loader
    test_img = single_input(img_path)
    # torchvision.utils.save_image(
    #     test_img, './fig/resnet18-features-input.png')
    single_img = test_img.unsqueeze(0)
    # print(single_img.shape)
    # assert False
    output = model(single_img)



if __name__ == '__main__':
    img_path = 'H:/TPAMI_channel_pruning/visualization/input_img/panda.jpeg'
    save_path = 'H:/TPAMI_channel_pruning/visualization'
    model_checkpoint_path = 'H:/TPAMI_channel_pruning/visualization/checkpoint/resnet18_p0.3.pth'
    model = ResNet(depth=depth, num_classes=n_classes)
    replace_layer_with_mask_conv_resnet(model)

    # load checkpoint
    check_point_params = torch.load(model_checkpoint_path, map_location='cpu')
    model_state = check_point_params["pruned_model"]
    model.load_state_dict(model_state)
    print("|===>load restrain file: {}".format(model_checkpoint_path))

    # set_cpu
    set_cpu()
    run(save_path, img_path, model)
