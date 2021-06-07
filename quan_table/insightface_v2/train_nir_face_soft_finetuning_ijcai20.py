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
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


from insightface_v2.model.models import ArcMarginModel
from insightface_v2.utils.utils import AverageMeter,  accuracy, get_logger, get_learning_rate, update_lr
from insightface_v2.utils.checkpoint import  save_checkpoint_soft
from insightface_v2.model_ijcai20.insightface_resnet import LResNet34E_IR
from insightface_v2.model_ijcai20.mobilefacenet import Mobilefacenet


def get_train_dataset(imgs_folder):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.501960784, 0.501960784, 0.501960784])
    ])  # H*W*C --> C*H*W, (X-127.5)/128 = (X/255.0-0.5)/0.501960784, 0.501960784=128/255.0
    ds = ImageFolder(imgs_folder, train_transform)
    # ds = ICCVImageFolder(imgs_folder, train_transform)

    # class_num = ds[-1][1] + 1
    class_num = len(ds.classes)
    return ds, class_num


def get_soft_finetune_train_loader(args):
    rgb_train_dataset, rgb_class_num = get_train_dataset(os.path.join(args.rgb_emore_folder, 'imgs'))
    nir_train_dataset, nir_class_num = get_train_dataset(os.path.join(args.nir_emore_folder, 'train_255'))
    # nir_train_dataset, nir_class_num = get_train_dataset(os.path.join(args.nir_emore_folder, 'train'))
    rgb_train_loader = DataLoader(dataset=rgb_train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
    nir_train_loader = DataLoader(dataset=nir_train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
    return rgb_train_loader, rgb_class_num, nir_train_loader, nir_class_num


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
    if args.network == 'Mobilefacenet':
        model = Mobilefacenet(embedding_size=args.emb_size)
    elif args.network == 'LResNet34E_IR':
        model = LResNet34E_IR(embedding_size=args.emb_size)
    else:
        assert False, logger.info('invalid network={}'.format(args.network))

    # logger.info(summary(model, (3, 112, 112)))
    rgb_metric_fc = ArcMarginModel(args, rgb_class_num, args.emb_size)
    nir_metric_fc = ArcMarginModel(args, nir_class_num, args.emb_size)

    # load pre-trained model
    if args.pretrained != '' :
        pretrained_state = torch.load(args.pretrained)
        model_state = pretrained_state["model"]

        if args.network == 'Mobilefacenet':
            rgb_metric_fc_state = pretrained_state["arcface"]
        elif args.network == 'LResNet34E_IR':
            rgb_metric_fc_state = pretrained_state["metric_fc"]
        else:
            assert False, logger.info('invalid network={}'.format(args.network))

        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
            logger.info("||===> load model finished !!!")

        if isinstance(rgb_metric_fc, nn.DataParallel):
            rgb_metric_fc.module.load_state_dict(rgb_metric_fc_state)
        else:
            rgb_metric_fc.load_state_dict(rgb_metric_fc_state)
            logger.info("||===> load rgb_metric_fc finished !!!")
        # logger.info("||===> load pretrained model finished !!!")


    # Initialize / load checkpoint
    if checkpoint is None or checkpoint == '':
        if args.optimizer == 'sgd':
            optimizer = optim.SGD([{'params': model.parameters()},
                                   {'params': rgb_metric_fc.parameters()},
                                   {'params': nir_metric_fc.parameters()}],
                                  lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay)
            logger.info('sgd optimizer={}'.format(optimizer))
        else:
            optimizer = optim.Adam([{'params': model.parameters()},
                                    {'params': rgb_metric_fc.parameters()},
                                    {'params': nir_metric_fc.parameters()}],
                                   lr=args.lr, weight_decay=args.weight_decay)
            logger.info('adam optimizer !!!')

        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.end_epoch)
        # logger.info('CosineAnnealingLR init !!!')
        scheduler = MultiStepLR(optimizer, milestones=args.step, gamma=args.gamma)
        logger.info('lr_scheduler: SGD MultiStepLR !!!')

    else:
        # to do
        assert False

    logger.info("model:{}".format(model))
    logger.info("rgb arcface:{}".format(rgb_metric_fc))
    logger.info("nir arcface:{}".format(nir_metric_fc))

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
    rgb_criterion = nn.CrossEntropyLoss().cuda()
    nir_criterion = nn.CrossEntropyLoss().cuda()
    logger.info("use CrossEntropyLoss !!!")


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
                save_checkpoint_soft(args, epoch, model, rgb_metric_fc, nir_metric_fc, optimizer, scheduler, nir_top1_accs)
        else:
            save_checkpoint_soft(args, epoch, model, rgb_metric_fc, nir_metric_fc, optimizer, scheduler, nir_top1_accs)


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

        # rgb_imgs, rgb_labels = next(iter(rgb_train_loader))
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
                'Epoch: [{:d}/{:d}][{:d}/{:d}]\t\tTrain: lr={:.5f}\trgb_loss={:.4f}'
                '\tnir_loss={:.4f}\ttotal_loss={:.4f}\trgb_Top1_Accuracy={:.4f}\tnir_Top1_Accuracy={:.4f}'
                .format(epoch, args.end_epoch, i, len(nir_train_loader), get_learning_rate(optimizer),
                        rgb_losses.avg, nir_losses.avg, total_losses.avg, rgb_top1_accs.avg, nir_top1_accs.avg))
    return rgb_losses.avg, nir_losses.avg, total_losses.avg, rgb_top1_accs.avg, nir_top1_accs.avg


# ijcai20 nir_soft_finetuning
def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')

    # # 241
    # parser.add_argument('--nir_emore_folder', default='/mnt/ssd/faces/nir_data/PolyU_NIR_Face_335', help='train data folder')
    # parser.add_argument('--rgb_emore_folder', default='/mnt/ssd/faces/faces_emore', help='train data folder')
    # parser.add_argument('--gpu', default='2', help='ngpu use')
    #
    # # Mobilefacenet
    # parser.add_argument('--pretrained', type=str, default="/home/xiezheng/program2019/insightface_DCP/insightface_v2/IJCAI20_Face_log/IJCAI20_face_pretrained_model/new_style_mobilefacenet_lfw_99.55_with_arcface_epoch34.pth", help='pretrained model')
    # parser.add_argument('--outpath', type=str, default='/home/xiezheng/program2019/insightface_DCP/insightface_v2/IJCAI20_Face_log/Mobilefacenet/', help='output path')
    # parser.add_argument('--network', default='mobilefacenet', help='specify network: r34, mobilefacenet')
    # parser.add_argument('--emb-size', type=int, default=128, help='embedding length')
    #
    # # LResNet34E_IR
    # # parser.add_argument('--pretrained', type=str, default="/home/xiezheng/program2019/insightface_DCP/insightface_v2/IJCAI20_Face_log/IJCAI20_face_pretrained_model/insightface_r34_with_arcface_v2_epoch48.pth", help='pretrained model')
    # # parser.add_argument('--outpath', type=str, default='/home/xiezheng/program2019/insightface_DCP/insightface_v2/IJCAI20_Face_log/LResNet34E_IR/', help='output path')
    # # parser.add_argument('--network', default='LResNet34E_IR', help='specify network: r34, mobilefacenet')
    # # parser.add_argument('--emb-size', type=int, default=512, help='embedding length')


    # # 245
    # parser.add_argument('--nir_emore_folder', default='/mnt/ssd/Datasets/Fine-Grained_Recognition/PolyU_NIR_Face_335',
    #                     help='train data folder')
    # parser.add_argument('--rgb_emore_folder', default='/mnt/ssd/Datasets/faces/faces_emore', help='train data folder')
    # parser.add_argument('--gpu', default='4', help='ngpu use')
    #
    # # Mobilefacenet
    # parser.add_argument('--pretrained', type=str,
    #                     default="/home/xiezheng/programs2019/insightface_DCP/insightface_v2/IJCAI20_Face_log/IJCAI20_face_pretrained_model/new_style_mobilefacenet_lfw_99.55_with_arcface_epoch34.pth",
    #                     help='pretrained model')
    # parser.add_argument('--outpath', type=str,
    #                     default='/home/xiezheng/programs2019/insightface_DCP/insightface_v2/IJCAI20_Face_log/Mobilefacenet/',
    #                     help='output path')
    # parser.add_argument('--network', default='mobilefacenet', help='specify network: r34, mobilefacenet')
    # parser.add_argument('--emb-size', type=int, default=128, help='embedding length')
    #
    # # LResNet34E_IR
    # # parser.add_argument('--pretrained', type=str,
    # #                     default="/home/xiezheng/programs2019/insightface_DCP/insightface_v2/IJCAI20_Face_log/IJCAI20_face_pretrained_model/insightface_r34_with_arcface_v2_epoch48.pth"
    # #                     ,help='pretrained model')
    # # parser.add_argument('--outpath', type=str,
    # #                     default='/home/xiezheng/programs2019/insightface_DCP/insightface_v2/IJCAI20_Face_log/LResNet34E_IR/', help='output path')
    # # parser.add_argument('--network', default='LResNet34E_IR', help='specify network: r34, mobilefacenet')
    # # parser.add_argument('--emb-size', type=int, default=512, help='embedding length')


    # 246
    parser.add_argument('--nir_emore_folder', default='/mnt/ssd/Datasets/Fine-Grained_Recognition/PolyU_NIR_Face_335',
                        help='train data folder')
    parser.add_argument('--rgb_emore_folder', default='/mnt/ssd/faces/faces_emore', help='train data folder')
    parser.add_argument('--gpu', default='1', help='ngpu use')

    # Mobilefacenet
    # parser.add_argument('--pretrained', type=str,
    #                     default="/home/xiezheng/program2019/insightface_DCP/insightface_v2/IJCAI20_Face_log/IJCAI20_face_pretrained_model/new_style_mobilefacenet_lfw_99.55_with_arcface_epoch34.pth",
    #                     help='pretrained model')
    # parser.add_argument('--outpath', type=str,
    #                     default='/home/xiezheng/program2019/insightface_DCP/insightface_v2/IJCAI20_Face_log/Mobilefacenet_nostep/',
    #                     help='output path')
    # parser.add_argument('--network', default='Mobilefacenet', help='specify network: LResNet34E_IR, Mobilefacenet')
    # parser.add_argument('--emb-size', type=int, default=128, help='embedding length')


    # LResNet34E_IR
    parser.add_argument('--pretrained', type=str,
                        default="/home/xiezheng/program2019/insightface_DCP/insightface_v2/IJCAI20_Face_log/IJCAI20_face_pretrained_model/insightface_r34_with_arcface_v2_epoch48.pth",
                        help='pretrained model')
    parser.add_argument('--outpath', type=str,
                        default='/home/xiezheng/program2019/insightface_DCP/insightface_v2/IJCAI20_Face_log/LResNet34E_IR_step/', help='output path')
    parser.add_argument('--network', default='LResNet34E_IR', help='specify network: LResNet34E_IR, Mobilefacenet')
    parser.add_argument('--emb-size', type=int, default=512, help='embedding length')


    parser.add_argument('--end-epoch', type=int, default=20, help='training epoch size.')
    parser.add_argument('--step', default=[13], type=list, help='step')
    parser.add_argument('--lr', type=float, default=0.0001, help='start learning rate')
    parser.add_argument('--cos_lr', type=bool, default=False, help='use cos_lr')
    parser.add_argument('--optimizer', default='sgd', help='optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size in each context: 512, 256')
    parser.add_argument('--margin-m', type=float, default=0.5, help='angular margin m')
    parser.add_argument('--margin-s', type=float, default=64.0, help='feature scale s')
    parser.add_argument('--easy-margin', type=bool, default=False, help='easy margin')
    parser.add_argument('--gamma', type=float, default=1.0, help='focusing parameter gamma')
    parser.add_argument('--use-se', type=bool, default=False, help='use SEBlock')
    parser.add_argument('--full-log', type=bool, default=False, help='full logging')
    parser.add_argument('--checkpoint', type=str, help='checkpoint')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
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
