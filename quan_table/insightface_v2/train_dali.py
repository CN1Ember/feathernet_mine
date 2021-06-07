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
# from tqdm import tqdm
# from prefetch_generator import BackgroundGenerator, background

from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from torchpie.parallel.reducer import reduce_tensor
from torchpie.experiment import distributed

from insightface_v2.insightface_data.data_pipe import get_train_loader, get_all_val_data, get_one_val_data
from insightface_v2.utils.verifacation import evaluate
from insightface_v2.model.focal_loss import FocalLoss
from insightface_v2.model.models import resnet18, resnet34, resnet50, resnet101, resnet152, \
    MobileNet, resnet_face18, ArcMarginModel, MobileFaceNet
from insightface_v2.utils.utils import parse_args, AverageMeter, clip_gradient, \
    accuracy, get_logger, get_learning_rate, separate_bn_paras, update_lr
from insightface_v2.utils.checkpoint import save_checkpoint, load_checkpoint

from insightface_v2.model.zq_mobilefacenet_old import ZQMobilefacenet
from insightface_v2.model.mobilenetv3 import MobileNetV3_Large

from insightface_v2.insightface_data.data_pipe_accimage import get_train_accimage_loader
from insightface_v2.model.face_model import face_model


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

def val_evaluate(args, model, carray, issame, nrof_folds = 5):
    model.eval()
    idx = 0
    embeddings = np.zeros([len(carray), args.emb_size])
    with torch.no_grad():
        while idx + args.batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx + args.batch_size])
            embeddings[idx:idx + args.batch_size] = l2_norm(model(batch.cuda()).cpu())  # xz: add l2_norm
            idx += args.batch_size
        if idx < len(carray):
            batch = torch.tensor(carray[idx:])
            embeddings[idx:] = l2_norm(model(batch.cuda()).cpu())   # xz: add l2_norm
    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    # buf = gen_plot(fpr, tpr)
    # roc_curve = Image.open(buf)
    # roc_curve_tensor = transforms.ToTensor()(roc_curve)
    return accuracy.mean(), best_thresholds.mean()

def board_val(args,logger, writer, db_name, accuracy, best_threshold, step):
    logger.info('||===>>Epoch: [[{:d}/{:d}]\t\tVal:{}_accuracy={:.4f}%\t\tbest_threshold={:.4f}'.format(step, args.end_epoch, db_name, accuracy*100, best_threshold))
    writer.add_scalar('val_{}_accuracy'.format(db_name), accuracy*100, step)
    writer.add_scalar('val_{}_best_threshold'.format(db_name), best_threshold, step)
    # writer.add_image('val_{}_roc_curve'.format(db_name), roc_curve_tensor, step)

def train_net(args):
    logger = get_logger(args.outpath, 'insightface')
    logger.info("args setting:\n{}".format(args))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    cudnn.benchmark = True

    checkpoint = args.checkpoint
    start_epoch = 0
    writer = SummaryWriter(args.outpath)
    lfw_best_acc = 0
    agedb_best_acc = 0
    cfp_best_acc = 0
    scheduler = None

    # train dataloader
    # train_loader, num_classes = get_train_loader(args)
    train_loader, num_classes = get_train_accimage_loader(args)

    logger.info("num_classes={:d}".format(num_classes))
    # validation dataset
    agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame = get_all_val_data(args.emore_folder)
    logger.info("train dataset and val dataset are ready!")

    # model setting
    face_metric_model = face_model(args, num_classes=num_classes)

    if args.pretrained != '' :
        pretrained_state = torch.load(args.pretrained)
        model_state = pretrained_state["model"]
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        logger.info("load pretrained model finished !!!")

    # Initialize / load checkpoint
    if checkpoint is None or checkpoint == '':
        if args.network in ['mobilefacenet_v1', 'mobilefacenet_v2', 'zq_mobilefacenet', 'mobilenet_v3']:
            paras_only_bn, paras_wo_bn = separate_bn_paras(model)
            if args.optimizer == 'sgd':
                logger.info("init mobilefacenet optimizer!!")
                optimizer = optim.SGD([{'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                                       {'params': [paras_wo_bn[-1]] + [metric_fc.weight], 'weight_decay': 4e-4},
                                       {'params': paras_only_bn}],
                                      lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay, nesterov=True)
        else:
            if args.optimizer == 'sgd':
                optimizer = optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                      lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay, nesterov=True)
            else:
                optimizer = optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                       lr=args.lr, weight_decay=args.weight_decay)

    else:
        # logger.info('load checkpoint!!!')
        model, metric_fc, start_epoch, optimizer, lfw_best_acc = load_checkpoint(checkpoint, model, metric_fc, logger)
        logger.info("load checkpoint : epoch ={}, lfw_best_acc={:4f}".format(start_epoch, lfw_best_acc))
        start_epoch = start_epoch + 1

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.end_epoch)

    model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
    if args.distributed:
        model = DDP(model, delay_allreduce=True)

    # Move to GPU, if available
    logger.info("{} model:{}".format(args.network, model))
    # logger.info(summary(model, (3, 112, 112)))
    logger.info("arcface:{}".format(metric_fc))

    # if torch.cuda.is_available():
    #     model = nn.DataParallel(model)
    #     metric_fc = nn.DataParallel(metric_fc)
    #
    #     model = model.cuda()
    #     metric_fc = metric_fc.cuda()

    # Loss function
    if args.focal_loss:
        if torch.cuda.is_available():
            criterion = FocalLoss(gamma=args.gamma).cuda()
        else:
            criterion = FocalLoss(gamma=args.gamma)
    else:
        if torch.cuda.is_available():
            criterion = nn.CrossEntropyLoss().cuda()
            logger.info("criterion = nn.CrossEntropyLoss().cuda()")
        else:
            criterion = nn.CrossEntropyLoss()
            logger.info("criterion = nn.CrossEntropyLoss()")

    logger.info("model, loss and optimizer are ready!")

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):

        # update_lr(epoch, optimizer, args)
        # start = datetime.now()

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        # One epoch's training
        train_loss, train_top1_accs, train_top5_accs= train(train_loader=train_loader, model=model, metric_fc=metric_fc,
                                            criterion=criterion, optimizer=optimizer, epoch=epoch, logger=logger, args=args)

        writer.add_scalar('Train_Loss', train_loss, epoch)
        writer.add_scalar('Train_lr', get_learning_rate(optimizer), epoch)
        writer.add_scalar('Train_Top1_Accuracy', train_top1_accs, epoch)
        writer.add_scalar('Train_Top5_Accuracy', train_top5_accs, epoch)

        # end = datetime.now()
        # delta = end - start
        # logger.info('{:.2f} seconds'.format(delta.seconds))

        # One epoch's validation
        agedb_accuracy, agedb_best_threshold = val_evaluate(args, model, agedb_30, agedb_30_issame)
        board_val(args,logger, writer, 'agedb_30', agedb_accuracy, agedb_best_threshold,  epoch)
        lfw_accuracy, lfw_best_threshold = val_evaluate(args, model, lfw, lfw_issame)
        board_val(args,logger, writer, 'lfw', lfw_accuracy, lfw_best_threshold, epoch)
        cfp_accuracy, cfp_best_threshold = val_evaluate(args, model, cfp_fp, cfp_fp_issame)
        board_val(args,logger, writer, 'cfp_fp', cfp_accuracy, cfp_best_threshold, epoch)

        # Check best acc
        is_best = (lfw_accuracy >= lfw_best_acc)

        lfw_best_acc = max(lfw_accuracy, lfw_best_acc)
        agedb_best_acc = max(agedb_accuracy, agedb_best_acc)
        cfp_best_acc = max(cfp_accuracy, cfp_best_acc)
        logger.info('||===>>Epoch: [{:d}/{:d}]\t\tlfw_best_acc={:.4f}\t\tagedb_best_acc={:.4f}\t\tcfp_best_acc={:.4f}\n'
                    .format(epoch, args.end_epoch, lfw_best_acc, agedb_best_acc, cfp_best_acc))

        # Save checkpoint
        if epoch <= args.end_epoch / 2:
            if epoch % 2 == 0:
                save_checkpoint(args, epoch, model, metric_fc, optimizer, scheduler, lfw_best_acc, is_best)
        else:
            save_checkpoint(args, epoch, model, metric_fc, optimizer, scheduler, lfw_best_acc, is_best)

        scheduler.step()


def train(train_loader, model, metric_fc, criterion, optimizer, epoch, logger, args):
    model.train()  # train mode (dropout and batchnorm is used)
    metric_fc.train()

    losses = AverageMeter()
    top1_accs = AverageMeter()
    top5_accs = AverageMeter()

    # use prefetch_generator and tqdm for iterating through data
    # pbar = tqdm(enumerate(BackgroundGenerator(train_loader)),total=len(train_loader))

    # Batches
    # for i, (img, label) in pbar:
    for i, data in enumerate(train_loader):
        # Move to GPU, if available

        # if  torch.cuda.is_available():
        #     img = img.cuda()
        #     label = label.cuda()  # [N, 1]
        img, label = data[0].cuda(non_blocking=True), data[1].cuda(non_blocking=True)

        # Forward prop.
        feature = model(img)  # embedding => [N, 512]
        output = metric_fc(feature, label)  # class_id_out => [N, 85742]

        # Calculate loss
        loss = criterion(output, label)

        # Back prop.
        optimizer.zero_grad()

        # loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        # Clip gradients
        clip_gradient(optimizer, args.grad_clip)

        # Update weights
        optimizer.step()

        top1_accuracy = accuracy(output, label, 1)
        top5_accuracy = accuracy(output, label, 5)

        if args.distributed:
            loss = reduce_tensor(loss)
            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)

        # Keep track of metrics
        losses.update(loss.item(), img.size(0))
        top1_accs.update(top1_accuracy, img.size(0))
        top5_accs.update(top5_accuracy, img.size(0))

        # Print status
        if i % args.print_freq == 0 :
            logger.info('Epoch: [{:d}/{:d}][{:d}/{:d}]\t\tTrain: lr={:.8f}\tLoss={:.4f}\tTop1 Accuracy={:.4f}\tTop5 Accuracy={:.4f}'
                        .format(epoch, args.end_epoch, i, len(train_loader), get_learning_rate(optimizer), losses.avg, top1_accs.avg, top5_accs.avg))

    return losses.avg, top1_accs.avg, top5_accs.avg


def main():
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
