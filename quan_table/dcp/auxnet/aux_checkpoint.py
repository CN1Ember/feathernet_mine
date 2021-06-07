import os

import torch
import torch.nn as nn

from dcp.checkpoint import CheckPoint


class AuxCheckPoint(CheckPoint):
    """
    save model state to file
    check_point_params: model, optimizer, epoch
    """

    def __init__(self, save_path, logger):
        super(AuxCheckPoint, self).__init__(save_path, logger)

    def save_aux_model(self, model, aux_fc):
        check_point_params = {}
        if isinstance(model, nn.DataParallel):
            check_point_params["model"] = model.module.state_dict()
        else:
            check_point_params["model"] = model.state_dict()
        aux_fc_state = []
        for i in range(len(aux_fc)):
            if isinstance(aux_fc[i], nn.DataParallel):
                aux_fc_state.append(aux_fc[i].module.state_dict())
            else:
                aux_fc_state.append(aux_fc[i].state_dict())
        check_point_params["aux_fc"] = aux_fc_state
        torch.save(check_point_params, os.path.join(self.save_path, "best_model_with_aux_fc.pth"))

    def save_aux_checkpoint(self, model, seg_optimizers, fc_optimizers, aux_fc, epoch, index=0):
        check_point_params = {}
        if isinstance(model, nn.DataParallel):
            check_point_params["model"] = model.module.state_dict()
        else:
            check_point_params["model"] = model.state_dict()
        seg_opt_state = []
        fc_opt_state = []
        aux_fc_state = []
        for i in range(len(seg_optimizers)):
            seg_opt_state.append(seg_optimizers[i].state_dict())
        for i in range(len(fc_optimizers)):
            fc_opt_state.append(fc_optimizers[i].state_dict())
            if isinstance(aux_fc[i], nn.DataParallel):
                aux_fc_state.append(aux_fc[i].module.state_dict())
            else:
                aux_fc_state.append(aux_fc[i].state_dict())

        check_point_params["seg_opt"] = seg_opt_state
        check_point_params["fc_opt"] = fc_opt_state
        check_point_params["aux_fc"] = aux_fc_state
        check_point_params['epoch'] = epoch

        torch.save(check_point_params, os.path.join(self.save_path, "checkpoint_{:0>3d}.pth".format(epoch)))


    ############################### insightface_dcp ######################################
    # save dcp_aux_model
    def face_save_aux_model(self, model, middle_layer, aux_fc, best_top1_acc):
        check_point_params = {}
        if isinstance(model, nn.DataParallel):
            check_point_params["model"] = model.module.state_dict()
        else:
            check_point_params["model"] = model.state_dict()

        middle_layer_state = []
        for i in range(len(middle_layer)):
            if isinstance(middle_layer[i], nn.DataParallel):
                middle_layer_state.append(middle_layer[i].module.state_dict())
            else:
                middle_layer_state.append(middle_layer[i].state_dict())
        check_point_params["middle_layer"] = middle_layer_state

        aux_fc_state = []
        for i in range(len(aux_fc)):
            if isinstance(aux_fc[i], nn.DataParallel):
                aux_fc_state.append(aux_fc[i].module.state_dict())
            else:
                aux_fc_state.append(aux_fc[i].state_dict())
        check_point_params["aux_fc"] = aux_fc_state
        torch.save(check_point_params, os.path.join(self.save_path,
                                                    "face_model_with_middle_aux_fc_lfw_{:.4f}.pth".format(best_top1_acc)))


    def face_save_aux_checkpoint(self, model, middle_layer, aux_fc, seg_optimizers, fc_optimizers, epoch):
        check_point_params = {}
        if isinstance(model, nn.DataParallel):
            check_point_params["model"] = model.module.state_dict()
        else:
            check_point_params["model"] = model.state_dict()

        middle_layer_state = []
        for i in range(len(middle_layer)):
            if isinstance(middle_layer[i], nn.DataParallel):
                middle_layer_state.append(middle_layer[i].module.state_dict())
            else:
                middle_layer_state.append(middle_layer[i].state_dict())

        aux_fc_state = []
        for i in range(len(aux_fc)):
            if isinstance(aux_fc[i], nn.DataParallel):
                aux_fc_state.append(aux_fc[i].module.state_dict())
            else:
                aux_fc_state.append(aux_fc[i].state_dict())

        seg_opt_state = []
        fc_opt_state = []
        for i in range(len(seg_optimizers)):
            seg_opt_state.append(seg_optimizers[i].state_dict())
        for i in range(len(fc_optimizers)):
            fc_opt_state.append(fc_optimizers[i].state_dict())

        check_point_params["seg_opt"] = seg_opt_state
        check_point_params["fc_opt"] = fc_opt_state
        check_point_params["aux_fc"] = aux_fc_state
        check_point_params["middle_layer"] = middle_layer_state
        check_point_params['epoch'] = epoch

        torch.save(check_point_params, os.path.join(self.save_path, "checkpoint_{:0>3d}.pth".format(epoch)))
