import os

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn

from prefetch_generator import BackgroundGenerator

from insightface_v2.utils.model_analyse import ModelAnalyse
from insightface_v2.utils.logger import get_logger
from insightface_v2.insightface_data.data_pipe import get_train_loader, get_all_val_data, get_test_loader
from insightface_v2.utils.verifacation import evaluate
from insightface_v2.model.focal_loss import FocalLoss
from insightface_v2.model.models import resnet34, resnet_face18, ArcMarginModel
from insightface_v2.utils.utils import parse_args, AverageMeter, clip_gradient, \
    accuracy, get_logger, get_learning_rate, separate_bn_paras, update_lr
from insightface_v2.utils.checkpoint import save_checkpoint, load_checkpoint
from insightface_v2.utils.ver_data import val_verification

from insightface_v2.model.zq_mobilefacenet import ZQMobilefacenet
# from insightface_v2.model.mobilefacenet_v3.mobilefacenetv3_v7 import MobileNetV3_v7

# from insightface_v2.model.mobilefacenet_v3.mobilefacenetv3_v4_depth import MobileNetV3_v4_depth

from insightface_v2.model.mobilefacenetv2.mobilefacenetv2 import Mobilefacenetv2
# from insightface_v2.model.mobilefacenetv2.mobilefacenet_hs import Mobilefacenet_hs
from insightface_v2.model.mobilefacenetv2.mobilefacenetv2_v3_width import Mobilefacenetv2_v3_width
from insightface_v2.model.mobilefacenetv2.mobilefacenetv2_v2_depth_width import Mobilefacenetv2_v2_depth
from insightface_v2.model.mobilefacenetv2.mobilefacenetv2_v4_width import Mobilefacenetv2_v4_width
from insightface_v2.model.mobilefacenetv2.mobilefacenetv2_v4_depth_width import Mobilefacenetv2_v4_depth_width

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

def val_evaluate(args, model, carray, issame, nrof_folds = 10):
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
    logger.info('||===>>Test Epoch: [[{:d}/{:d}]\t\tVal:{}_accuracy={:.4f}%\t\tbest_threshold={:.4f}'
                .format(step, args.end_epoch, db_name, accuracy*100, best_threshold))
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
    tpr_best = 0

    scheduler = None

    # train dataloader
    train_loader, num_classes = get_train_loader(args)
    test_loader = get_test_loader(args)
    logger.info("num_classes={:d}".format(num_classes))
    # validation dataset
    agedb_30, cfp_fp, lfw, verfication, agedb_30_issame, cfp_fp_issame, lfw_issame, verfication_issame \
        = get_all_val_data(args.emore_folder)
    logger.info("train dataset and val dataset are ready!")


    # model setting
    if args.network == 'r34':
        model = resnet34(args)
    elif args.network == 'mobilefacenet_v1':
        # model = MobileFaceNet(embedding_size=args.emb_size, blocks=args.block_setting)  # mobilefacenet_v1

        model = Mobilefacenetv2(embedding_size=args.emb_size)

    elif args.network == 'mobilefacenet_v2':

        model = Mobilefacenetv2_v2_depth(embedding_size=args.emb_size, blocks=args.block_setting, fc_type=args.fc_type)
        # model = Mobilefacenetv2_v2_depth(blocks=[3, 8, 16, 5], embedding_size=512, fc_type="gdc")

        # model = Mobilefacenetv2_v3_width(embedding_size=args.emb_size, width_mult=args.width_mult)

        # model = Mobilefacenetv2_v4_width(blocks=args.block_setting, embedding_size=args.emb_size,
        # fc_type=args.fc_type, width_mult=args.width_mult)
        # model = Mobilefacenetv2_v4_width(blocks=[4, 6, 2], embedding_size=512, fc_type="gdc", width_mult=1.312)

        # model = Mobilefacenetv2_v4_depth_width(blocks=args.block_setting, embedding_size=args.emb_size,
        #                                        fc_type=args.fc_type, width_mult=args.width_mult)
        # model = Mobilefacenetv2_v4_depth_width(blocks=[2, 8, 16, 8], embedding_size=512, fc_type="gdc", width_mult=1.0)

        # model = Mobilefacenetv2_v4_depth_width(blocks=args.block_setting,
        #                                        embedding_size=args.emb_size, fc_type=args.fc_type, width_mult=args.width_mult)
        # Mobilefacenetv2_v4_depth_width(blocks=[2, 6, 9, 5], embedding_size=512, fc_type="gdc", width_mult=1.1)


    elif args.network == 'zq_mobilefacenet':
        model = ZQMobilefacenet(embedding_size=args.emb_size, blocks=args.block_setting, fc_type=args.fc_type)

    else:
        model = resnet_face18(args.use_se)

    # logger.info(summary(model, (3, 112, 112)))
    if args.loss_type in ["arcface"]:
        metric_fc = ArcMarginModel(args, num_classes, args.emb_size)
    elif args.loss_type in ["softmax"]:
        metric_fc = nn.Linear(args.emb_size, num_classes)
    else:
        assert False

    if args.pretrained != '' :
        pretrained_state = torch.load(args.pretrained)
        model_state = pretrained_state["model"]
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        # metric_fc_state = pretrained_state["metric_fc"]
        logger.info("load pretrained model finished !!!")


    # Initialize / load checkpoint
    if checkpoint is None or checkpoint == '':
        if args.network in ['mobilefacenet_v1', 'mobilefacenet_v2', 'zq_mobilefacenet', 'mobilenet_v3']:
            paras_only_bn, paras_wo_bn, paras_bias = separate_bn_paras(model)
            if args.optimizer == 'sgd':
                logger.info("init mobilefacenet optimizer!!")
                optimizer = optim.SGD([{'params': paras_wo_bn, 'weight_decay': 4e-5},
                                       {'params': metric_fc.weight, 'weight_decay': 4e-4},
                                       {'params': paras_only_bn, 'weight_decay': 0},
                                       {'params': paras_bias, 'lr': 2 * args.lr, 'weight_decay': 0, 'is_bias': True}],
                                      lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay, nesterov=True)
        else:
            if args.optimizer == 'sgd':
                optimizer = optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                      lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay, nesterov=True)
            else:
                optimizer = optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                       lr=args.lr, weight_decay=args.weight_decay)
        if args.cos_lr:
            logger.info("CosineAnnealingLR !!!")
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.end_epoch)

    else:
        # logger.info('load checkpoint!!!')
        # model, metric_fc, start_epoch, optimizer, scheduler, lfw_best_acc = \
        #     load_checkpoint(checkpoint, model, metric_fc, logger)

        model, metric_fc, start_epoch, optimizer, lfw_best_acc = load_checkpoint(checkpoint, model, metric_fc, logger)
        logger.info("load checkpoint : epoch ={}, lfw_best_acc={:4f}".format(start_epoch, lfw_best_acc))
        start_epoch = start_epoch + 1
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.end_epoch)

    # Move to GPU, if available
    logger.info("{} model:{}".format(args.network, model))
    # logger.info(summary(model, (3, 112, 112)))
    logger.info("metric_fc:{}".format(metric_fc))

    save_path = './finetune-test'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    test_input = torch.randn(1, 3, 112, 112)
    model_analyse = ModelAnalyse(model, logger)

    params_num = model_analyse.params_count()
    flops = model_analyse.flops_compute(test_input)
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            count = count + 1

    logger.info("\nmodel layers_num = {}".format(count))
    logger.info("model size={} MB".format(params_num * 4 / 1024 / 1024))
    logger.info("model flops={} M\n".format(sum(flops) / (10 ** 6)))

    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        metric_fc = nn.DataParallel(metric_fc)

        model = model.cuda()
        metric_fc = metric_fc.cuda()

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
        # start = datetime.now()
        # if args.cos_lr:
        #     scheduler.step(epoch)
        # else:
        #     update_lr(epoch, optimizer, args)

        # One epoch's training
        train_loss, train_top1_accs, train_top5_accs= train(train_loader=train_loader, model=model, metric_fc=metric_fc,
                                            criterion=criterion, optimizer=optimizer, epoch=epoch, logger=logger, args=args)
        test_loss, test_top1_acc, test_top5_acc = test(test_loader=test_loader, model=model, metric_fc=metric_fc,
                                                        criterion=criterion,epoch=epoch, logger=logger, args=args)

        writer.add_scalar('Train_Loss', train_loss, epoch)
        writer.add_scalar('Train_lr', get_learning_rate(optimizer), epoch)
        writer.add_scalar('Train_Top1_Accuracy', train_top1_accs, epoch)
        writer.add_scalar('Train_Top5_Accuracy', train_top5_accs, epoch)

        writer.add_scalar('Test_Loss', test_loss, epoch)
        writer.add_scalar('Test_Top1_Accuracy', test_top1_acc, epoch)
        writer.add_scalar('Test_Top5_Accuracy', test_top5_acc, epoch)
        logger.info("||===> Test Epoch:[[{:d}/{:d}]\t\t Test_Loss:{:.4f}\t\tTest_Top1_Accuracy:{:.4f}%\t\tTest_Top5_Accuracy:{:.4f}%"
                    .format(epoch, args.end_epoch, test_loss, test_top1_acc*100, test_top5_acc*100))

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

        tpr = val_verification(args, model, verfication, verfication_issame, fpr=10 ** -3)
        logger.info('||===>>Test Epoch: [[{:d}/{:d}]\t\tVal: FAR=10e-3, TPR_accuracy={:.4f}%'
                    .format(epoch, args.end_epoch, float(tpr)*100))
        writer.add_scalar('Test_TPR', float(tpr)*100, epoch)

        # Check best acc
        is_best = (lfw_accuracy >= tpr_best)

        tpr_best = max(lfw_accuracy, tpr_best)
        lfw_best_acc = max(lfw_accuracy, lfw_best_acc)
        agedb_best_acc = max(agedb_accuracy, agedb_best_acc)
        cfp_best_acc = max(cfp_accuracy, cfp_best_acc)
        logger.info('||===>>Epoch: [{:d}/{:d}]\t\tlfw_best_acc={:.4f}%\t\tagedb_best_acc={:.4f}%\t\t'
                    'cfp_best_acc={:.4f}%\t\tFAR=10e-3, TPR_best={:.4f}\n\n'
                    .format(epoch, args.end_epoch, lfw_best_acc*100, agedb_best_acc*100, cfp_best_acc*100, tpr_best*100))

        # Save checkpoint
        if epoch <= args.end_epoch / 2:
            if epoch % 2 == 0:
                save_checkpoint(args, epoch, model, metric_fc, optimizer, scheduler, tpr_best, is_best)
        else:
            save_checkpoint(args, epoch, model, metric_fc, optimizer, scheduler, tpr_best, is_best)

        # if epoch <= args.end_epoch / 2:
        #     if epoch % 2 == 0:
        #         save_checkpoint(args, epoch, model, metric_fc, optimizer, lfw_best_acc, is_best)
        # else:
        #     save_checkpoint(args, epoch, model, metric_fc, optimizer, lfw_best_acc, is_best)

        if args.cos_lr:
            scheduler.step()
        else:
            update_lr(epoch, optimizer, args)

def train(train_loader, model, metric_fc, criterion, optimizer, epoch, logger, args):
    model.train()  # train mode (dropout and batchnorm is used)
    metric_fc.train()

    losses = AverageMeter()
    top1_accs = AverageMeter()
    top5_accs = AverageMeter()

    # use prefetch_generator and tqdm for iterating through data
    pbar = enumerate(BackgroundGenerator(train_loader))

    # Batches
    for i, (img, label) in pbar:
    # for i, (img, label) in enumerate(train_loader):
        # Move to GPU, if available
        if  torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()  # [N, 1]

        # Forward prop.
        feature = model(img)  # embedding => [N, 512]

        if args.loss_type in ["arcface"]:
            output = metric_fc(feature, label)  # class_id_out => [N, 85742]
        elif args.loss_type in ["softmax"]:
            output = metric_fc(feature)
        else:
            assert False

        # Calculate loss
        loss = criterion(output, label)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, args.grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item(), img.size(0))

        top1_accuracy = accuracy(output, label, 1)
        top1_accs.update(top1_accuracy, img.size(0))
        top5_accuracy = accuracy(output, label, 5)
        top5_accs.update(top5_accuracy, img.size(0))

        # Print status
        if i % args.print_freq == 0 :
            logger.info('Epoch: [{:d}/{:d}][{:d}/{:d}]\t\tTrain: lr={:.8f}\tLoss={:.4f}\tTop1 Accuracy={:.4f}%\tTop5 Accuracy={:.4f}%'
                        .format(epoch, args.end_epoch, i, len(train_loader),
                                get_learning_rate(optimizer), losses.avg, top1_accs.avg*100, top5_accs.avg*100))
        # break
    return losses.avg, top1_accs.avg, top5_accs.avg


def test(test_loader, model, metric_fc, criterion, epoch, logger, args):
    model.eval()  # train mode (dropout and batchnorm is used)
    metric_fc.eval()

    losses = AverageMeter()
    top1_accs = AverageMeter()
    top5_accs = AverageMeter()

    # use prefetch_generator and tqdm for iterating through data
    pbar = enumerate(BackgroundGenerator(test_loader))

    # Batches
    for i, (img, label) in pbar:
    # for i, (img, label) in enumerate(train_loader):
        # Move to GPU, if available
        if  torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()  # [N, 1]

        # Forward prop.
        feature = model(img)  # embedding => [N, 512]

        if args.loss_type in ["arcface"]:
            output = metric_fc(feature, label)  # class_id_out => [N, 85742]
        elif args.loss_type in ["softmax"]:
            output = metric_fc(feature)
        else:
            assert False

        # Calculate loss
        loss = criterion(output, label)

        # Keep track of metrics
        losses.update(loss.item(), img.size(0))

        top1_accuracy = accuracy(output, label, 1)
        top1_accs.update(top1_accuracy, img.size(0))
        top5_accuracy = accuracy(output, label, 5)
        top5_accs.update(top5_accuracy, img.size(0))

        # Print status
        if i % args.print_freq == 0 :
            logger.info('Test Epoch: [{:d}/{:d}]\tLoss={:.4f}\tTop1 Accuracy={:.4f}%\tTop5 Accuracy={:.4f}%'
                        .format(i, len(test_loader), losses.avg, top1_accs.avg*100, top5_accs.avg*100))
        # break
    return losses.avg, top1_accs.avg, top5_accs.avg


def main():
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
