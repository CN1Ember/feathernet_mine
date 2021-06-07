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
import torch.nn.functional as F

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


def loss_kl(outputs, teacher_outputs, T=2.0):
    kl_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1),
                                                  F.softmax(teacher_outputs / T, dim=1)) * (T * T)
    return kl_loss


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
    logger.info("train dataset and val dataset are ready!")

    # model setting
    if args.network == 'Mobilefacenet':
        source_model = Mobilefacenet(embedding_size=args.emb_size)
        target_model = Mobilefacenet(embedding_size=args.emb_size)
    elif args.network == 'LResNet34E_IR':
        source_model = LResNet34E_IR(embedding_size=args.emb_size)
        target_model = LResNet34E_IR(embedding_size=args.emb_size)
    else:
        assert False, logger.info('invalid network={}'.format(args.network))

    # logger.info(summary(model, (3, 112, 112)))
    source_metric_fc = ArcMarginModel(args, rgb_class_num, args.emb_size)
    target_metric_fc = ArcMarginModel(args, nir_class_num, args.emb_size)

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

        if isinstance(source_model, nn.DataParallel):
            source_model.module.load_state_dict(model_state)
            target_model.module.load_state_dict(model_state)
        else:
            source_model.load_state_dict(model_state)
            target_model.load_state_dict(model_state)
            logger.info("||===> load model finished !!!")

        if isinstance(source_metric_fc, nn.DataParallel):
            source_metric_fc.module.load_state_dict(rgb_metric_fc_state)
        else:
            source_metric_fc.load_state_dict(rgb_metric_fc_state)
            logger.info("||===> load rgb_metric_fc finished !!!")
        # logger.info("||===> load pretrained model finished !!!")


    # Initialize / load checkpoint
    if checkpoint is None or checkpoint == '':
        if args.optimizer == 'sgd':
            optimizer = optim.SGD([{'params': target_model.parameters()},
                                   {'params': source_metric_fc.parameters()},
                                   {'params': target_metric_fc.parameters()}],
                                  lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay)
            logger.info('sgd optimizer={}'.format(optimizer))
        else:
            assert False

        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.end_epoch)
        # logger.info('CosineAnnealingLR init !!!')
        scheduler = MultiStepLR(optimizer, milestones=args.step, gamma=args.gamma)
        logger.info('lr_scheduler: SGD MultiStepLR !!!')

    else:
        # to do
        assert False

    logger.info("target_model:{}".format(target_model))
    logger.info("source_metric_fc:{}".format(source_metric_fc))
    logger.info("target_metric_fc:{}".format(target_metric_fc))

    # DataParallel
    if torch.cuda.is_available():
        # source_model = nn.DataParallel(source_model)
        # target_model = nn.DataParallel(target_model)
        #
        # source_metric_fc = nn.DataParallel(source_metric_fc)
        # target_metric_fc = nn.DataParallel(target_metric_fc)

        source_model = source_model.cuda()
        target_model = target_model.cuda()

        source_metric_fc = source_metric_fc.cuda()
        target_metric_fc = target_metric_fc.cuda()
        logger.info("model to cuda")

    # Loss function
    # source_criterion = nn.CrossEntropyLoss().cuda()
    target_criterion = nn.CrossEntropyLoss().cuda()
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
        rgb_loss, nir_loss, total_loss, nir_top1_accs= \
            train(nir_train_loader, source_model, source_metric_fc, target_model, target_metric_fc, target_criterion,
                  optimizer=optimizer, epoch=epoch, logger=logger, args=args)

        logger.info("train return finish !!!")
        writer.add_scalar('rgb_loss', rgb_loss, epoch)
        writer.add_scalar('nir_loss', nir_loss, epoch)
        writer.add_scalar('total_loss', total_loss, epoch)
        writer.add_scalar('Train_lr', get_learning_rate(optimizer), epoch)
        writer.add_scalar('nir_top1_accs', nir_top1_accs, epoch)
        # logger.info("writer add_scalar finished !!!")

        end = datetime.now()
        delta = end - start
        logger.info('{:.2f} seconds'.format(delta.seconds))

        # Save checkpoint
        if epoch <= args.end_epoch / 2:
            if epoch % 2 == 0:
                save_checkpoint_soft(args, epoch, target_model, source_metric_fc, target_metric_fc, optimizer, scheduler, nir_top1_accs)
        else:
            save_checkpoint_soft(args, epoch, target_model, source_metric_fc, target_metric_fc, optimizer, scheduler, nir_top1_accs)



def train(nir_train_loader, source_model, source_metric_fc, target_model, target_metric_fc, target_criterion,
          optimizer, epoch, logger, args):

    # train mode (dropout and batchnorm is used)
    source_model.eval()
    source_metric_fc.train()
    target_model.train()
    target_metric_fc.train()

    kl_losses = AverageMeter()
    clc_losses = AverageMeter()
    total_losses = AverageMeter()
    nir_top1_accs = AverageMeter()


    # Batches
    # for i, (img, label) in enumerate(train_loader):
    for i, (nir_imgs, nir_labels) in enumerate(nir_train_loader):
        # Move to GPU, if available

        nir_imgs = nir_imgs.cuda()
        nir_labels = nir_labels.cuda()  # [N, 1]

        # Forward prop.
        with torch.no_grad():
            source_feature = source_model(nir_imgs)

        source_output = source_metric_fc(source_feature, nir_labels)

        target_feature = target_model(nir_imgs)
        target_source_output = source_metric_fc(target_feature, nir_labels)
        target_output = target_metric_fc(target_feature, nir_labels)

        # print(source_output.shape)
        # print(target_source_output.shape)
        # print(source_output-target_source_output)
        # assert False
        kl_loss = loss_kl(target_source_output, source_output, T=2.0)
        clc_loss = target_criterion(target_output, nir_labels)

        # combine loss
        total_loss = 1000*kl_loss + clc_loss

        # Back prop.
        optimizer.zero_grad()
        total_loss.backward()

        # Update weights
        optimizer.step()

        # Keep track of metrics
        kl_losses.update(kl_loss.item(), nir_imgs.size(0))
        clc_losses.update(clc_loss.item(), nir_imgs.size(0))
        total_losses.update(total_loss.item(), nir_imgs.size(0))

        nir_top1_accuracy = accuracy(target_output, nir_labels, 1)
        nir_top1_accs.update(nir_top1_accuracy, nir_imgs.size(0))


        # Print status
        if i % args.print_freq == 0:
            logger.info(
                'Epoch: [{:d}/{:d}][{:d}/{:d}]\t\tTrain: lr={:.5f}\tkl_loss*1000={:.4f}'
                '\tclc_loss={:.4f}\ttotal_loss={:.4f}\tnir_Top1_Accuracy={:.4f}'
                .format(epoch, args.end_epoch, i, len(nir_train_loader), get_learning_rate(optimizer),
                        kl_losses.avg*1000, clc_losses.avg, total_losses.avg, nir_top1_accs.avg))

    return kl_losses.avg, clc_losses.avg, total_losses.avg, nir_top1_accs.avg


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
    parser.add_argument('--gpu', default='2', help='ngpu use')

    # Mobilefacenet
    # parser.add_argument('--pretrained', type=str,
    #                     default="/home/xiezheng/program2019/insightface_DCP/insightface_v2/IJCAI20_Face_log/IJCAI20_face_pretrained_model/new_style_mobilefacenet_lfw_99.55_with_arcface_epoch34.pth",
    #                     help='pretrained model')
    # parser.add_argument('--outpath', type=str,
    #                     default='/home/xiezheng/program2019/insightface_DCP/insightface_v2/IJCAI20_Face_log/'
    #                             'lwf/Mobilefacenet_lr0.0001/',
    #                     help='output path')
    # parser.add_argument('--network', default='Mobilefacenet', help='specify network: LResNet34E_IR, Mobilefacenet')
    # parser.add_argument('--emb-size', type=int, default=128, help='embedding length')


    # LResNet34E_IR
    parser.add_argument('--pretrained', type=str,
                        default="/home/xiezheng/program2019/insightface_DCP/insightface_v2/IJCAI20_Face_log/IJCAI20_face_pretrained_model/insightface_r34_with_arcface_v2_epoch48.pth",
                        help='pretrained model')
    parser.add_argument('--outpath', type=str,
                        default='/home/xiezheng/program2019/insightface_DCP/insightface_v2/IJCAI20_Face_log/'
                                'lwf/LResNet34E_IR_lr0.0001/', help='output path')
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
