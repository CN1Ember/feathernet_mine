import sys

sys.path.append("../insightface_v2")

import mxnet
import torch
from insightface_v2.model.mobilefacenetv2.mobilefacenetv2_v2_depth import Mobilefacenetv2_v2_depth


def _convert_bn(a):
    print(a)
    return 1, 2


dict = {"bn.running_mean": "batchnorm_moving_mean",
        "bn.running_var": "batchnorm_moving_var",
        "prelu.weight": "relu_gamma",
        "conv1": "conv_sep",
        "conv2": "conv_dw",
        "conv3": "conv_proj"
        }


def convert_from_mxnet(model, metric_fc, checkpoint_prefix):
    _, mxnet_weights, mxnet_aux = mxnet.model.load_checkpoint(checkpoint_prefix, 0)
    key_w = mxnet_weights.keys()
    key_a = mxnet_aux.keys()
    key_a = list(key_a)
    key_w = list(key_w)
    len_key = len(key_w)
    len_key_a = len(key_a)

    # for i in key_w:
    #     print(i)
    #     print(mxnet_weights.get(i).asnumpy())
    #     # break

    # print(mxnet_weights)
    # exit(1)
    # print(mxnet_aux.get(i).asnumpy().shape)

    # key = list(key)
    # for i in key:
    #     # print(i)
    #     print(mxnet_weights.get(i).asnumpy().shape)
    #     # print(mxnet_aux.get(i).asnumpy().shape)
    # exit(1)

    remapped_state = {}
    i = 0
    # mxnet_aux
    moving_mean_i = 0
    moving_var_i = 1

    # mxnet_weights
    conv_weight_i = 0
    batchnorm_gamma_i = 1
    batchnorm_beta_i = 2
    relu_gamma_i = 3
    liniear_i = 396
    new_dict = {}
    for state_key in model.state_dict().keys():
        k = state_key.split('.')
        # print(state_key)
        # print(model.state_dict().get(state_key).shape)
        # continue
        # aux = False
        # mxnet_key = ''
        # last
        last_key = k[len(k) - 1]
        # laset second key
        lsec_key = k[len(k) - 2]

        if last_key == "num_batches_tracked":
            continue
        if last_key == "running_mean":
            # if moving_mean_i>=len_key_a:
            #     break
            value = mxnet_aux.get(key_a[moving_mean_i])
            value = value.asnumpy()
            new_dict[state_key] = torch.from_numpy(value)
            moving_mean_i += 2

        if last_key == "running_var":
            # if moving_var_i>=len_key_a:
            #     break
            value = mxnet_aux.get(key_a[moving_var_i])
            value = value.asnumpy()
            new_dict[state_key] = torch.from_numpy(value)
            moving_var_i += 2

        if last_key == "weight":
            if lsec_key.find("conv") != -1:
                value = mxnet_weights.get(key_w[conv_weight_i])
                value = value.asnumpy()
                # print(value)
                # print(torch.from_numpy(value).double())
                # break
                new_dict[state_key] = torch.from_numpy(value)

                while True:
                    conv_weight_i += 1
                    if conv_weight_i >= len_key:
                        break
                    if key_w[conv_weight_i].find("conv") != -1 and key_w[conv_weight_i].find("weight") != -1:
                        break

            if lsec_key.find("bn") != -1:
                value = mxnet_weights.get(key_w[batchnorm_gamma_i])
                value = value.asnumpy()
                new_dict[state_key] = torch.from_numpy(value)
                while True:
                    batchnorm_gamma_i += 1
                    if batchnorm_gamma_i >= len_key:
                        break
                    if key_w[batchnorm_gamma_i].find("batchnorm") != -1 and key_w[batchnorm_gamma_i].find(
                            "gamma") != -1:
                        # liniear_i = batchnorm_gamma_i - 2
                        break

            if lsec_key.find("prelu") != -1:
                value = mxnet_weights.get(key_w[relu_gamma_i])
                value = value.asnumpy()
                new_dict[state_key] = torch.from_numpy(value)
                while True:
                    relu_gamma_i += 1
                    # print(relu_gamma_i)
                    # print(len_key)
                    if relu_gamma_i >= len_key:
                        break
                    if key_w[relu_gamma_i].find("relu") != -1 and key_w[relu_gamma_i].find("gamma") != -1:
                        break

            if lsec_key.find("linear") != -1:
                # print("weight")
                # print(liniear_i)
                # print(len_key)
                value = mxnet_weights.get(key_w[liniear_i])
                # print(value.shape)
                value = value.asnumpy()
                new_dict[state_key] = torch.from_numpy(value)
                liniear_i+=1
                # while True:
                #     liniear_i += 1
                #     if liniear_i >= len_key:
                #         break
                #     if key_w[liniear_i].find("fc1") != -1 and key_w[liniear_i].find("weight") != -1:
                #         break

            # if lsec_key.find("conv") == -1 and lsec_key.find("bn") == -1 and lsec_key.find("prelu") == -1:
            #     value = mxnet_weights.get(key_w[batchnorm_gamma_i])
            #     value = value.asnumpy()
            #     new_dict.update([state_key,[torch.from_numpy(value)]])

        if last_key == "bias":
            if lsec_key.find("bn") != -1:
                value = mxnet_weights.get(key_w[batchnorm_beta_i])
                value = value.asnumpy()
                new_dict[state_key] = torch.from_numpy(value)
                while True:
                    batchnorm_beta_i += 1
                    if batchnorm_beta_i >= len_key:
                        break
                    if key_w[batchnorm_beta_i].find("batchnorm") != -1 and key_w[batchnorm_beta_i].find("beta") != -1:
                        # liniear_i = batchnorm_gamma_i - 2
                        break
            if lsec_key.find("linear") != -1:
                # print("bias")
                # print(liniear_i)
                # print(len_key)
                value = mxnet_weights.get(key_w[liniear_i])
                # print(value.shape)
                value = value.asnumpy()
                new_dict[state_key] = torch.from_numpy(value)
                # while True:
                #     liniear_i += 1
                #     if liniear_i >= len_key:
                #         break
                #     if key_w[liniear_i].find("fc1") != -1 and key_w[liniear_i].find("bias") != -1:
                #         break
        # i += 1
        # if i == 28:
        #     break

        # if k[0] == 'features':
        #     if k[1] == 'conv1_1':
        #         # input block
        #         mxnet_key += 'conv1_x_1__'
        #         if k[2] == 'bn':
        #             mxnet_key += 'relu-sp__bn_'
        #             aux, key_add = _convert_bn(k[3])
        #             mxnet_key += key_add
        #         else:
        #             assert k[3] == 'weight'
        #             mxnet_key += 'conv_' + k[3]
        #     elif k[1] == 'conv5_bn_ac':
        #         # bn + ac at end of features block
        #         mxnet_key += 'conv5_x_x__relu-sp__bn_'
        #         assert k[2] == 'bn'
        #         aux, key_add = _convert_bn(k[3])
        #         mxnet_key += key_add
        #     else:
        #         # middle blocks
        #         if model.b and 'c1x1_c' in k[2]:
        #             bc_block = True  # b-variant split c-block special treatment
        #         else:
        #             bc_block = False
        #         ck = k[1].split('_')
        #         mxnet_key += ck[0] + '_x__' + ck[1] + '_'
        #         ck = k[2].split('_')
        #         mxnet_key += ck[0] + '-' + ck[1]
        #         if ck[1] == 'w' and len(ck) > 2:
        #             mxnet_key += '(s/2)' if ck[2] == 's2' else '(s/1)'
        #         mxnet_key += '__'
        #         if k[3] == 'bn':
        #             mxnet_key += 'bn_' if bc_block else 'bn__bn_'
        #             aux, key_add = _convert_bn(k[4])
        #             mxnet_key += key_add
        #         else:
        #             ki = 3 if bc_block else 4
        #             assert k[ki] == 'weight'
        #             mxnet_key += 'conv_' + k[ki]
        # elif k[0] == 'classifier':
        #     if 'fc6-1k_weight' in mxnet_weights:
        #         mxnet_key += 'fc6-1k_'
        #     else:
        #         mxnet_key += 'fc6_'
        #     mxnet_key += k[1]
        # else:
        #     assert False, 'Unexpected token'
        #
        # if debug:
        #     print(mxnet_key, '=> ', state_key, end=' ')
        #
        # mxnet_array = mxnet_aux[mxnet_key] if aux else mxnet_weights[mxnet_key]
        # torch_tensor = torch.from_numpy(mxnet_array.asnumpy())
        # if k[0] == 'classifier' and k[1] == 'weight':
        #     torch_tensor = torch_tensor.view(torch_tensor.size() + (1, 1))
        # remapped_state[state_key] = torch_tensor
        #
        # if debug:
        #     print(list(torch_tensor.size()), torch_tensor.mean(), torch_tensor.std())
    metric_dic = {}
    for state_key in metric_fc.state_dict().keys():
        if state_key == "weight":
            value = mxnet_weights.get(key_w[len(key_w) - 2])
            value = value.asnumpy()
            metric_dic[state_key] = torch.from_numpy(value)
        elif state_key == "bias":
            value = mxnet_weights.get(key_w[len(key_w) - 1])
            value = value.asnumpy()
            metric_dic[state_key] = torch.from_numpy(value)

    # print(new_dict)
    # exit(1)
    model.load_state_dict(new_dict)
    metric_fc.load_state_dict(metric_dic)
    # print(model.state_dict())
    return model, metric_fc


if __name__ == '__main__':
    model = Mobilefacenetv2_v2_depth(blocks=[3, 8, 16, 5], embedding_size=512, fc_type="gdc")
    metric_fc = torch.nn.Linear(512, 93431)
    # print(model)
    # exit(1)
    convert_from_mxnet(model, metric_fc, "/home/xiezheng/insightface/test_models/MobileFaceNet")
