import os
from datetime import datetime
from shutil import copyfile

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
import torch.backends.cudnn as cudnn
from PIL import Image
from torchsummary import summary
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator, background
import argparse

from insightface_v2.insightface_data.data_pipe import get_soft_finetune_train_loader
from insightface_v2.utils.verifacation import evaluate
from insightface_v2.model.focal_loss import FocalLoss
from insightface_v2.model.models import resnet18, resnet34, resnet50, resnet101, resnet152, \
     MobileNet, resnet_face18, ArcMarginModel, MobileFaceNet
from insightface_v2.utils.utils import AverageMeter, clip_gradient, accuracy, \
    get_logger, get_learning_rate, separate_bn_paras, update_lr
from insightface_v2.utils.checkpoint import save_checkpoint, load_checkpoint, save_checkpoint_soft

from insightface_v2.model.mobilefacenet_pruned import pruned_Mobilefacenet
from insightface_v2.model.mobilefacenetv2_width_wm import Mobilefacenetv2_width_wm
from insightface_v2.model.cross_entropy_label_smooth import CrossEntropyLabelSmooth

from insightface_v2.ncnn_quantization import get_activation_value, replace


def board_val(args,logger, writer, db_name, accuracy, best_threshold, step):
    logger.info('||===>>Epoch: [[{:d}/{:d}]\t\tVal:{}_accuracy={:.4f}%\t\tbest_threshold={:.4f}'.format(step, args.end_epoch, db_name, accuracy*100, best_threshold))
    writer.add_scalar('val_{}_accuracy'.format(db_name), accuracy*100, step)
    # writer.add_scalar('val_{}_best_threshold'.format(db_name), best_threshold, step)
    # writer.add_image('val_{}_roc_curve'.format(db_name), roc_curve_tensor, step)


def load_state(model, state_dict, logger):
    """
            load state_dict to model
            :params model:
            :params state_dict:
            :return: model
            """
    model.eval()
    model_dict = model.state_dict()

    for key, value in list(state_dict.items()):
        if key in list(model_dict.keys()):
            logger.info('load key={}'.format(key))
            if key == 'conv3.weight':
                logger.info('break !!!')
                break
            model_dict[key] = value
        else:
            if logger:
                logger.error("key error: {} {}".format(key, value.size))
            # assert False
    model.load_state_dict(model_dict)
    return model


def train_net(args):
    logger = get_logger(args.outpath, 'nir_face')
    logger.info("args setting:\n{}".format(args))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    cudnn.benchmark = True

    checkpoint = args.checkpoint
    start_epoch = 0
    writer = SummaryWriter(args.outpath)
    val_nir_best_acc = 0
    scheduler = None

    # train dataloader
    rgb_train_loader, rgb_class_num, nir_train_loader, nir_class_num = get_soft_finetune_train_loader(args)
    logger.info("rgb_class_num={}, nir_class_num={}".format(rgb_class_num, nir_class_num))
    # validation dataset
    # val_nir, val_nir_issame = get_one_val_data(args.emore_folder, 'nir_val_aligned')
    # val_nir, val_nir_issame = get_one_val_data(args.emore_folder, 'nir_face_test_400_verfication_data')
    logger.info("train dataset and val dataset are ready!")

    # model setting
    if args.network == 'mobilefacenet_p0.5':
        # model = pruned_Mobilefacenet(pruning_rate=0.5)   # old
        model = Mobilefacenetv2_width_wm(embedding_size=args.emb_size, pruning_rate=0.5)
    else:
        assert False

    rgb_metric_fc = ArcMarginModel(args, rgb_class_num, args.emb_size)
    nir_metric_fc = ArcMarginModel(args, nir_class_num, args.emb_size)

    # load pretrain_model and rgb_aux_fc
    if args.pretrained != '' :
        pretrained_state = torch.load(args.pretrained)
        model_state = pretrained_state["model"]
        rgb_metric_fc_state = pretrained_state["aux_fc"]

        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)

        # if isinstance(rgb_metric_fc, nn.DataParallel):
        #     logger.info('rgb_metric_fc_state[-1] shape={}'.format(rgb_metric_fc_state[-1].shape))
        #     rgb_metric_fc.module.load_state_dict(rgb_metric_fc_state[-1])
        # else:
        #     rgb_metric_fc.load_state_dict(rgb_metric_fc_state[-1])

        if isinstance(rgb_metric_fc, nn.DataParallel):
            # logger.info('rgb_metric_fc_state[-1] shape={}'.format(rgb_metric_fc_state.shape))
            rgb_metric_fc.module.load_state_dict(rgb_metric_fc_state)
        else:
            rgb_metric_fc.load_state_dict(rgb_metric_fc_state)

        # model = load_state(model, model_state, logger)
        logger.info("||===> load pretrained model and rgb_aux_fc finished !!!")

    logger.info("model:{}".format(model))
    logger.info("rgb arcface:{}".format(rgb_metric_fc))
    logger.info("nir arcface:{}".format(nir_metric_fc))

    # replace model
    activation_value_list = get_activation_value(args.table_path)
    model = replace(model, activation_value_list, logger)
    logger.info("After replace:\n {}".format(model))


    # Initialize / load checkpoint
    if checkpoint is None or checkpoint == '':
        if args.optimizer == 'sgd':
            optimizer = optim.SGD([{'params': model.parameters()},
                                   {'params': rgb_metric_fc.parameters()},
                                   {'params': nir_metric_fc.parameters()}],
                                  lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay, nesterov=True)
            logger.info('sgd optimizer={}'.format(optimizer))
        else:
            optimizer = optim.Adam([{'params': model.parameters()},
                                    {'params': rgb_metric_fc.parameters()},
                                    {'params': nir_metric_fc.parameters()}],
                                   lr=args.lr, weight_decay=args.weight_decay)
            logger.info('adam optimizer !!!')

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.end_epoch)
        logger.info('CosineAnnealingLR init !!!')

    else:
        # to do
        # logger.info('load checkpoint!!!')
        # model, metric_fc, start_epoch, optimizer, lfw_best_acc = load_checkpoint(checkpoint, model, metric_fc, logger)
        # logger.info("load checkpoint : epoch ={}, lfw_best_acc={:4f}".format(start_epoch, lfw_best_acc))
        assert False

    # DataParallel
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        rgb_metric_fc = nn.DataParallel(rgb_metric_fc)
        nir_metric_fc = nn.DataParallel(nir_metric_fc)

        model = model.cuda()
        rgb_metric_fc = rgb_metric_fc.cuda()
        nir_metric_fc = nir_metric_fc.cuda()
        logger.info("model to cuda")

    # Loss function
    if args.use_label_smooth:
        rgb_criterion = CrossEntropyLabelSmooth(num_classes=rgb_class_num, epsilon=0.1)
        nir_criterion = CrossEntropyLabelSmooth(num_classes=nir_class_num, epsilon=0.1)
        logger.info("use CrossEntropyLabelSmooth !!!")
    else:
        assert False

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        if args.cos_lr:
            scheduler.step(epoch)
            logger.info("CosineAnnealingLR update !!!")
        else:
            update_lr(epoch, optimizer, args)
            logger.info("step_lr update !!!")

        start = datetime.now()
        # One epoch's training
        rgb_loss, nir_loss, total_loss, rgb_top1_accs, nir_top1_accs= \
            train(rgb_train_loader=rgb_train_loader, nir_train_loader=nir_train_loader, model=model,
                  rgb_metric_fc=rgb_metric_fc, nir_metric_fc=nir_metric_fc,
                  rgb_criterion=rgb_criterion,nir_criterion=nir_criterion,
                  optimizer=optimizer, epoch=epoch, logger=logger, args=args)

        logger.info("train return finish !!!")
        writer.add_scalar('rgb_loss', rgb_loss, epoch)
        writer.add_scalar('nir_loss', nir_loss, epoch)
        writer.add_scalar('total_loss', total_loss, epoch)
        writer.add_scalar('Train_lr', get_learning_rate(optimizer), epoch)
        writer.add_scalar('rgb_top1_accs', rgb_top1_accs, epoch)
        writer.add_scalar('nir_top1_accs', nir_top1_accs, epoch)
        # logger.info("writer add_scalar finished !!!")

        end = datetime.now()
        delta = end - start
        logger.info('{:.2f} seconds'.format(delta.seconds))

        # One epoch's validation
        # val_nir_accuracy, val_nir_best_threshold = val_evaluate(args, model, val_nir, val_nir_issame)
        # # logger.info("val return finish !!!")
        # board_val(args, logger, writer, 'val_nir', val_nir_accuracy, val_nir_best_threshold, epoch)
        # # logger.info("print val result !!!")
        #
        # # agedb_accuracy, agedb_best_threshold = val_evaluate(args, model, agedb_30, agedb_30_issame)
        # # board_val(args, logger, writer, 'agedb_30', agedb_accuracy, agedb_best_threshold, epoch)
        # # lfw_accuracy, lfw_best_threshold = val_evaluate(args, model, lfw, lfw_issame)
        # # board_val(args, logger, writer, 'lfw', lfw_accuracy, lfw_best_threshold, epoch)
        # # cfp_accuracy, cfp_best_threshold = val_evaluate(args, model, cfp_fp, cfp_fp_issame)
        # # board_val(args, logger, writer, 'cfp_fp', cfp_accuracy, cfp_best_threshold, epoch)
        #
        # # Check best acc
        # # is_best = (val_nir_accuracy >= val_nir_best_acc)
        # val_nir_best_acc = max(val_nir_accuracy, val_nir_best_acc)
        # logger.info('||===>>Epoch: [{:d}/{:d}]\t\tval_nir_best_acc={:.4f}\n'.format(epoch, args.end_epoch, val_nir_best_acc))

        # Save checkpoint
        if epoch <= args.end_epoch / 2:
            if epoch % 2 == 0:
                save_checkpoint_soft(args, epoch, model, rgb_metric_fc, nir_metric_fc, optimizer, scheduler, val_nir_best_acc)
        else:
            save_checkpoint_soft(args, epoch, model, rgb_metric_fc, nir_metric_fc, optimizer, scheduler, val_nir_best_acc)


def train(rgb_train_loader, nir_train_loader, model, rgb_metric_fc, nir_metric_fc, rgb_criterion, nir_criterion,
          optimizer, epoch, logger, args):

    # train mode (dropout and batchnorm is used)
    model.train()
    rgb_metric_fc.train()
    nir_metric_fc.train()

    rgb_losses = AverageMeter()
    nir_losses = AverageMeter()
    total_losses = AverageMeter()
    rgb_top1_accs = AverageMeter()
    nir_top1_accs = AverageMeter()

    # use prefetch_generator and tqdm for iterating through data
    rgb_pbar = BackgroundGenerator(rgb_train_loader)

    alpha = min(1, epoch / (args.end_epoch - 1))
    logger.info('soft finetuning alpha={}'.format(alpha))

    # Batches
    # for i, (img, label) in enumerate(train_loader):
    for i, (nir_imgs, nir_labels) in enumerate(nir_train_loader):
        # Move to GPU, if available

        nir_imgs = nir_imgs.cuda()
        nir_labels = nir_labels.cuda()  # [N, 1]

        rgb_imgs, rgb_labels = rgb_pbar.next()

        rgb_imgs = rgb_imgs.cuda()
        rgb_labels = rgb_labels.cuda()

        # Forward prop.
        nir_feature = model(nir_imgs)  # embedding => [N, 512]
        nir_output = nir_metric_fc(nir_feature, nir_labels)  # class_id_out => [N, 85742]
        # Calculate loss
        nir_loss = nir_criterion(nir_output, nir_labels)

        rgb_feature = model(rgb_imgs)  # embedding => [N, 512]
        rgb_output = rgb_metric_fc(rgb_feature, rgb_labels)  # class_id_out => [N, 85742]
        rgb_loss = rgb_criterion(rgb_output, rgb_labels)

        # combine loss
        total_loss = (1.0 - alpha)*rgb_loss + nir_loss

        # Back prop.
        optimizer.zero_grad()
        total_loss.backward()

        # Update weights
        optimizer.step()

        # Keep track of metrics
        rgb_losses.update(rgb_loss.item(), rgb_imgs.size(0))
        nir_losses.update(nir_loss.item(), nir_imgs.size(0))
        total_losses.update(total_loss.item(), nir_imgs.size(0))

        nir_top1_accuracy = accuracy(nir_output, nir_labels, 1)
        nir_top1_accs.update(nir_top1_accuracy, nir_imgs.size(0))

        rgb_top1_accuracy = accuracy(rgb_output, rgb_labels, 1)
        rgb_top1_accs.update(rgb_top1_accuracy, rgb_imgs.size(0))

        # Print status
        if i % args.print_freq == 0:
            logger.info(
                'Epoch: [{:d}/{:d}][{:d}/{:d}]\t\tTrain: lr={:.6f}\trgb_loss={:.4f}'
                '\tnir_loss={:.4f}\ttotal_loss={:.4f}\trgb_Top1_Accuracy={:.4f}\tnir_Top1_Accuracy={:.4f}'
                .format(epoch, args.end_epoch, i, len(nir_train_loader), get_learning_rate(optimizer),
                        rgb_losses.avg, nir_losses.avg, total_losses.avg, rgb_top1_accs.avg, nir_top1_accs.avg))
    return rgb_losses.avg, nir_losses.avg, total_losses.avg, rgb_top1_accs.avg, nir_top1_accs.avg


# insightface mobilefacenet_v1_our_sq + nir_soft_finetuning_qqc_sq_int8: nir_data_1321: follow_liujing
def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    # 1224
    # parser.add_argument('--nir_data_mode', default='nir_face_1224', help='train_data mode: nir_face_1224')
    # 1321
    # parser.add_argument('--nir_data_mode', default='nir_face_1321_align',
    #                     help='train_data mode: nir_face_1224, nir_face_1321_align, nir_face_1420_align')
    # 1420
    parser.add_argument('--nir_data_mode', default='nir_face_1420_align',
                        help='train_data mode: nir_face_1224, nir_face_1321_align, nir_face_1420_align')


    parser.add_argument('--rgb_data_mode', default='iccv_emore',
                        help='train_data mode: iccv_emore')
    # 241
    # parser.add_argument('--nir_emore_folder', default='/mnt/ssd/faces/nir_data', help='train data folder')
    # parser.add_argument('--rgb_emore_folder', default='/mnt/ssd/faces/iccv_challenge/train/ms1m-retinaface-t1', help='train data folder')

    # 246
    parser.add_argument('--nir_emore_folder', default='/mnt/ssd/nir_face_data', help='train data folder')
    parser.add_argument('--rgb_emore_folder', default='/mnt/ssd/faces/iccv_challenge/train/ms1m-retinaface-t1',
                        help='train data folder')
    parser.add_argument('--gpu', default='6', help='ngpu use')

    # 1224
    # parser.add_argument('--outpath', type=str,
    #                     default='/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/'
    #                             '2stage/dim128/mf_p0.5_without_p_fc_nir_soft_finetune_lr0.0001/'
    #                             'mf_p0.5_without_p_fc_qqc_sq_ckp11_then_nir_soft_finetune_1224_epoch26_qqc_sq_nostep', help='output path')
    # 1321
    # parser.add_argument('--outpath', type=str,
    #                     default='/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/'
    #                             '2stage/dim128/mf_p0.5_without_p_fc_nir_soft_finetune_lr0.0001/'
    #                             'mf_p0.5_without_p_fc_qqc_sq_ckp11_then_nir_soft_finetune_1321_epoch26_qqc_sq_nostep', help='output path')
    # 1420
    parser.add_argument('--outpath', type=str,
                        default='/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/'
                                '2stage/dim128/mf_p0.5_without_p_fc_nir_soft_finetune_lr0.0001/'
                                'mf_p0.5_without_p_fc_qqc_sq_ckp11_then_nir_soft_finetune_1420_epoch26_qqc_sq_nostep', help='output path')


    # mobilefacent_p0.5_iccv_new_without_p_fc_our_sq: checkpoint_17
    # parser.add_argument('--pretrained', type=str, default="/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/quantization/log_aux_mobilefacenetv2_baseline_0.5width_without_fc_128_arcface_iccv_emore_bs384_e18_lr0.001_step[]_without_fc_cosine_quantization_finetune_20190811/check_point/checkpoint_017.pth",
    #                     help='pretrained model')

    # mobilefacent_p0.5_iccv_new_without_p_fc_our_sq_checkpoint_17_then_qqc_sq_chp_11
    parser.add_argument('--pretrained', type=str,
                        default="/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/qqc_sq_finetune/log_ft_mobilefacenet_v1_p0.5_iccv_ms1m_bs256_e12_lr0.000_step[]_p0.5_our_sq_then_qqc_sq_finetune_20190913/check_point/checkpoint_011.pth",
                        help='pretrained model')
    parser.add_argument('--table_path', type=str,
                        default="/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage"
                                "/dim128/qqc_sq_finetune/iccv_mobilefacenet_p0.5_our_sq_activation.table", help='pretrained model')

    # parser.add_argument('--network', default='mobilefacenet_v1', help='specify network: r34, mobilefacenet')
    parser.add_argument('--network', default='mobilefacenet_p0.5', help='specify network: r34, mobilefacenet')

    parser.add_argument('--block_setting', default=[1,4,6,2], type=list, help='block num')
    parser.add_argument('--last_type', type=str, default='soft-finetuning', help='part update type')

    parser.add_argument('--end-epoch', type=int, default=26, help='training epoch size.')

    # parser.add_argument('--step', default=[20], type=list, help='step')
    # parser.add_argument('--step', default=[10, 20], type=list, help='step')

    parser.add_argument('--step', default=[], type=list, help='step')
    parser.add_argument('--cos_lr', type=bool, default=False, help='use cos_lr')

    parser.add_argument('--lr', type=float, default=0.0001, help='start learning rate')
    # parser.add_argument('--lr', type=float, default=0.001, help='start learning rate')

    parser.add_argument('--optimizer', default='sgd', help='optimizer')
    parser.add_argument('--weight-decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')

    parser.add_argument('--emb-size', type=int, default=128, help='embedding length')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size in each context: 512, 256')

    parser.add_argument('--margin-m', type=float, default=0.5, help='angular margin m')
    parser.add_argument('--margin-s', type=float, default=64.0, help='feature scale s')
    parser.add_argument('--easy-margin', type=bool, default=False, help='easy margin')

    parser.add_argument('--focal-loss', type=bool, default=False, help='focal loss')
    parser.add_argument('--gamma', type=float, default=2.0, help='focusing parameter gamma')
    parser.add_argument('--use-se', type=bool, default=False, help='use SEBlock')
    parser.add_argument('--full-log', type=bool, default=False, help='full logging')

    parser.add_argument('--use_label_smooth', type=bool, default=True, help='use label_smooth')
    parser.add_argument('--checkpoint', type=str, help='checkpoint')
    parser.add_argument('--num_workers', type=int, default=8, help='num workers')
    parser.add_argument('--print_freq', type=int, default=10, help='print freq')

    args = parser.parse_args()
    # ensure_folder(args.outpath)
    if not os.path.isdir(args.outpath):
        os.makedirs(args.outpath)
    return args


def main():
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
