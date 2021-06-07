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

from insightface_v2.insightface_data.data_pipe import get_train_loader, get_all_val_data, get_one_val_data, get_val_pair
from insightface_v2.utils.verifacation import evaluate
from insightface_v2.model.focal_loss import FocalLoss
from insightface_v2.model.models import resnet18, resnet34, resnet50, resnet101, resnet152, \
     MobileNet, resnet_face18, ArcMarginModel, MobileFaceNet
from insightface_v2.utils.utils import parse_args, AverageMeter, clip_gradient, accuracy, \
    get_logger, get_learning_rate, separate_bn_paras, update_lr
from insightface_v2.utils.checkpoint import save_checkpoint, load_checkpoint

from insightface_v2.model.mobilefacenet_pruned import pruned_Mobilefacenet
from insightface_v2.model.mobilefacenetv2_width_wm import Mobilefacenetv2_width_wm

# def full_log(epoch):
#     full_log_dir = 'data/full_log'
#     if not os.path.isdir(full_log_dir):
#         os.mkdir(full_log_dir)
#     filename = 'angles_{}.txt'.format(epoch)
#     dst_file = os.path.join(full_log_dir, filename)
#     src_file = 'data/angles.txt'
#     copyfile(src_file, dst_file)

# def l2_norm(input, axis=1):
#     norm = torch.norm(input, 2, axis, True)
#     output = torch.div(input, norm)
#     return output
#
# def val_evaluate(args, model, carray, issame, nrof_folds = 10):
#     model.eval()
#     idx = 0
#     embeddings = np.zeros([len(carray), args.emb_size])
#     with torch.no_grad():
#         while idx + args.batch_size <= len(carray):
#             batch = torch.tensor(carray[idx:idx + args.batch_size])
#             embeddings[idx:idx + args.batch_size] = l2_norm(model(batch.cuda()).cpu())  # xz: add l2_norm
#             idx += args.batch_size
#         if idx < len(carray):
#             batch = torch.tensor(carray[idx:])
#             embeddings[idx:] = l2_norm(model(batch.cuda()).cpu())   # xz: add l2_norm
#     tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
#     # buf = gen_plot(fpr, tpr)
#     # roc_curve = Image.open(buf)
#     # roc_curve_tensor = transforms.ToTensor()(roc_curve)
#     return accuracy.mean(), best_thresholds.mean()

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
    train_loader, num_classes = get_train_loader(args)
    logger.info("num_classes={:d}".format(num_classes))
    # validation dataset
    # val_nir, val_nir_issame = get_one_val_data(args.emore_folder, 'nir_val_aligned')
    # val_nir, val_nir_issame = get_one_val_data(args.emore_folder, 'nir_face_test_400_verfication_data')
    logger.info("train dataset and val dataset are ready!")

    # model setting
    if args.network == 'r34':
        model = resnet34(args)
    elif args.network == 'mobilefacenet_v1':
        model = MobileFaceNet(args.emb_size, blocks=args.block_setting)  # mobilefacenet_v1
    elif args.network == 'mobilefacenet_v2':
        model = MobileFaceNet(args.emb_size, blocks=args.block_setting)  # mobilefacenet_v2

    elif args.network == 'mobilefacenet_p0.5':
        # model = pruned_Mobilefacenet(pruning_rate=0.5)   # old
        model = Mobilefacenetv2_width_wm(embedding_size=args.emb_size, pruning_rate=0.5)

    else:
        model = resnet_face18(args.use_se)

    # logger.info(summary(model, (3, 112, 112)))
    metric_fc = ArcMarginModel(args, num_classes, args.emb_size)

    if args.pretrained != '' :
        pretrained_state = torch.load(args.pretrained)
        model_state = pretrained_state["model"]

        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)

        # model = load_state(model, model_state, logger)

        logger.info("||===> load pretrained model finished !!!")

    # Initialize / load checkpoint
    if checkpoint is None or checkpoint == '':
        if args.network in ['mobilefacenet_v1', 'mobilefacenet_v2', 'mobilefacenet_p0.5']:
            if args.optimizer == 'sgd':
                # optimizer = optim.SGD([{'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                #                        {'params': [paras_wo_bn[-1]] + [metric_fc.weight], 'weight_decay': 4e-4},
                #                        {'params': paras_only_bn}],
                #                       lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay, nesterov=True)

                if args.network == 'mobilefacenet_v1':
                    # [last_1 + arcface] update
                    # logger.info('[last_1 + arcface] update')
                    # optimizer = optim.SGD([{'params': model.linear.parameters()},
                    #                        {'params': model.bn.parameters()},
                    #                        {'params': metric_fc.parameters()}],
                    #                       lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay, nesterov=True)

                    # [last_2 + arcface] update
                    # logger.info('[last_2 + arcface] update')
                    # optimizer = optim.SGD([{'params': model.conv_6_dw.parameters()},
                    #                        {'params': model.linear.parameters()},
                    #                        {'params': model.bn.parameters()},
                    #                        {'params': metric_fc.parameters()}],
                    #                       lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay, nesterov=True)

                    # [last_3 + arcface] update
                    logger.info('[last_3 + arcface] update')
                    optimizer = optim.SGD([{'params': model.conv_6_sep.parameters(), 'weight_decay': 4e-5},
                                           {'params': model.conv_6_dw.parameters()},
                                           {'params': model.linear.parameters()},
                                           {'params': model.bn.parameters()},
                                           {'params': metric_fc.parameters()}],
                                          lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay, nesterov=True)

                elif args.network == 'mobilefacenet_p0.5':

                     if args.last_type == 'last1':
                         logger.info('mobilefacenet_p0.5: [last_1 + arcface] update')
                         # [last_1 + arcface] update
                         optimizer = optim.SGD([{'params': model.linear.parameters()},
                                                {'params': model.bn5.parameters()},
                                                {'params': metric_fc.parameters()}],
                                               lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay,
                                               nesterov=True)
                     elif args.last_type == 'last2':
                         logger.info('mobilefacenet_p0.5: [last_2 + arcface] update')
                         # [last_2 + arcface] update
                         optimizer = optim.SGD([{'params': model.conv4.parameters()},
                                            {'params': model.bn4.parameters()},
                                            {'params': model.linear.parameters()},
                                            {'params': model.bn5.parameters()},
                                            {'params': metric_fc.parameters()}],
                                           lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay, nesterov=True)
                     elif args.last_type == 'last3':
                         logger.info('mobilefacenet_p0.5: [last_3 + arcface] update')
                         # [last_3 + arcface] update
                         optimizer = optim.SGD([{'params': model.conv3.parameters(), 'weight_decay': 4e-5},
                                                {'params': model.bn3.parameters(), 'weight_decay': 4e-5},
                                                {'params': model.prelu3.parameters(), 'weight_decay': 4e-5},

                                                {'params': model.conv4.parameters()},
                                                {'params': model.bn4.parameters()},
                                                {'params': model.linear.parameters()},
                                                {'params': model.bn5.parameters()},
                                                {'params': metric_fc.parameters()}],
                                               lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay,
                                               nesterov=True)
                     elif args.last_type == 'last4':
                         logger.info('mobilefacenet_p0.5: [last_4 + arcface] update')
                         # [last_3 + arcface] update
                         logger.info("model.layers[-1].conv3={}".format(model.layers[-1].conv3))
                         logger.info("model.layers[-1].bn3={}".format(model.layers[-1].bn3))

                         optimizer = optim.SGD([{'params': model.layers[-1].conv3.parameters(), 'weight_decay': 4e-5},
                                                {'params': model.layers[-1].bn3.parameters(), 'weight_decay': 4e-5},

                                                {'params': model.conv3.parameters(), 'weight_decay': 4e-5},
                                                {'params': model.bn3.parameters(), 'weight_decay': 4e-5},
                                                {'params': model.prelu3.parameters(), 'weight_decay': 4e-5},

                                                {'params': model.conv4.parameters()},
                                                {'params': model.bn4.parameters()},
                                                {'params': model.linear.parameters()},
                                                {'params': model.bn5.parameters()},
                                                {'params': metric_fc.parameters()}],
                                               lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay,
                                               nesterov=True)
                     elif args.last_type == 'last5':
                         logger.info('mobilefacenet_p0.5: [last_5 + arcface] update')
                         # [last_3 + arcface] update
                         optimizer = optim.SGD([{'params': model.layers[-1].conv2.parameters(), 'weight_decay': 4e-5},
                                                {'params': model.layers[-1].bn2.parameters(), 'weight_decay': 4e-5},
                                                {'params': model.layers[-1].prelu2.parameters(), 'weight_decay': 4e-5},
                                                {'params': model.layers[-1].conv3.parameters(), 'weight_decay': 4e-5},
                                                {'params': model.layers[-1].bn3.parameters(), 'weight_decay': 4e-5},

                                                {'params': model.conv3.parameters(), 'weight_decay': 4e-5},
                                                {'params': model.bn3.parameters(), 'weight_decay': 4e-5},
                                                {'params': model.prelu3.parameters(), 'weight_decay': 4e-5},

                                                {'params': model.conv4.parameters()},
                                                {'params': model.bn4.parameters()},
                                                {'params': model.linear.parameters()},
                                                {'params': model.bn5.parameters()},
                                                {'params': metric_fc.parameters()}],
                                               lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay,
                                               nesterov=True)
                     elif args.last_type == 'last6':
                         logger.info('mobilefacenet_p0.5: [last_6 + arcface] update')
                         # [last_3 + arcface] update
                         logger.info('model.layers[-1]={}'.format(model.layers[-1]))

                         optimizer = optim.SGD([{'params': model.layers[-1].parameters(), 'weight_decay': 4e-5},
                                                {'params': model.conv3.parameters(), 'weight_decay': 4e-5},
                                                {'params': model.bn3.parameters(), 'weight_decay': 4e-5},
                                                {'params': model.prelu3.parameters(), 'weight_decay': 4e-5},

                                                {'params': model.conv4.parameters()},
                                                {'params': model.bn4.parameters()},
                                                {'params': model.linear.parameters()},
                                                {'params': model.bn5.parameters()},
                                                {'params': metric_fc.parameters()}],
                                               lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay,
                                               nesterov=True)
                     else:
                         assert False, logger.info('not support last_type:{}'.format(args.last_type))

                logger.info("load mobilefacenet part optimizer!!")

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

    # Move to GPU, if available
    logger.info("model:{}".format(model))
    logger.info("arcface:{}".format(metric_fc))

    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        metric_fc = nn.DataParallel(metric_fc)

        model = model.cuda()
        metric_fc = metric_fc.cuda()
        logger.info("model to cuda")

    # Loss function
    if args.focal_loss:
        if torch.cuda.is_available():
            criterion = FocalLoss(gamma=args.gamma).cuda()
        else:
            criterion = FocalLoss(gamma=args.gamma)
    else:
        if torch.cuda.is_available():
            criterion = nn.CrossEntropyLoss().cuda()
        else:
            criterion = nn.CrossEntropyLoss()

    # logger.info("model, loss and optimizer are ready!")
    # val_nir_accuracy, _ = val_evaluate(args, model, val_nir, val_nir_issame)
    # logger.info("before training val_accuracy = {:.4f}".format(val_nir_accuracy))

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        update_lr(epoch, optimizer, args)
        start = datetime.now()
        # One epoch's training
        train_loss, train_top1_accs, train_top5_accs= train(train_loader=train_loader, model=model, metric_fc=metric_fc,
                                            criterion=criterion, optimizer=optimizer, epoch=epoch, logger=logger, args=args)
        logger.info("train return finish !!!")
        writer.add_scalar('Train_Loss', train_loss, epoch)
        writer.add_scalar('Train_lr', get_learning_rate(optimizer), epoch)
        writer.add_scalar('Train_Top1_Accuracy', train_top1_accs, epoch)
        writer.add_scalar('Train_Top5_Accuracy', train_top5_accs, epoch)
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
                save_checkpoint(args, epoch, model, metric_fc, optimizer, scheduler, val_nir_best_acc)
        else:
            save_checkpoint(args, epoch, model, metric_fc, optimizer, scheduler, val_nir_best_acc)



def train(train_loader, model, metric_fc, criterion, optimizer, epoch, logger, args):
    # train mode (dropout and batchnorm is used)
    # model.train()
    # metric_fc.train()

    logger.info("args.network={}".format(args.network))
    if args.network == 'mobilefacenet_v1':
        # [last_1 + arcface] update
        # model.eval()
        # model.module.bn.train()  # focus
        # metric_fc.train()
        # logger.info('[last_1 + arcface] update')

        # [last_2 + arcface] update
        # model.eval()
        # model.module.bn.train()  # focus
        # model.module.conv_6_dw.bn.train()
        # metric_fc.train()
        # logger.info('[last_2 + arcface] update')

        # [last_3 + arcface] update
        model.eval()
        model.module.bn.train()  # focus
        model.module.conv_6_dw.bn.train()
        model.module.conv_6_sep.bn.train()
        metric_fc.train()
        logger.info("[last_3 + arcface] update")

    elif args.network == 'mobilefacenet_p0.5':
        if args.last_type == 'last1':
            # [last_1 + arcface] update
            model.eval()
            model.module.bn5.train()  # focus
            metric_fc.train()
            logger.info("[last_1 + arcface] update")

        elif args.last_type == 'last2':
            # [last_2 + arcface] update
            model.eval()
            model.module.bn4.train()
            model.module.bn5.train()  # focus
            metric_fc.train()
            logger.info('[last_2 + arcface] update')

        elif args.last_type == 'last3':
            # [last_3 + arcface] update
            model.eval()
            model.module.bn3.train()
            model.module.bn4.train()
            model.module.bn5.train()  # focus
            metric_fc.train()
            logger.info('[last_3 + arcface] update')

        elif args.last_type == 'last4':
            # [last_3 + arcface] update
            model.eval()
            model.module.layers[-1].bn3.train()
            model.module.bn3.train()
            model.module.bn4.train()
            model.module.bn5.train()  # focus
            metric_fc.train()
            logger.info('[last_4 + arcface] update')

        elif args.last_type == 'last5':
            # [last_3 + arcface] update
            model.eval()
            model.module.layers[-1].bn2.train()
            model.module.layers[-1].bn3.train()
            model.module.bn3.train()
            model.module.bn4.train()
            model.module.bn5.train()  # focus
            metric_fc.train()
            logger.info('[last_5 + arcface] update')

        elif args.last_type == 'last6':
            # [last_3 + arcface] update
            model.eval()
            model.module.layers[-1].bn1.train()
            model.module.layers[-1].bn2.train()
            model.module.layers[-1].bn3.train()
            model.module.bn3.train()
            model.module.bn4.train()
            model.module.bn5.train()  # focus
            metric_fc.train()
            logger.info('[last_6 + arcface] update')

        else:
            assert False, logger.info('not support last_type:{}'.format(args.last_type))


    losses = AverageMeter()
    top1_accs = AverageMeter()
    top5_accs = AverageMeter()

    # use prefetch_generator and tqdm for iterating through data
    pbar = enumerate(BackgroundGenerator(train_loader))

    # Batches
    # for i, (img, label) in enumerate(train_loader):
    for i, (img, label) in pbar:
        # Move to GPU, if available
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()  # [N, 1]

        # Forward prop.
        feature = model(img)  # embedding => [N, 512]
        output = metric_fc(feature, label)  # class_id_out => [N, 85742]

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
        # if i % args.print_freq == 0:
        logger.info('Epoch: [{:d}/{:d}][{:d}/{:d}]\t\tTrain: lr={:.5f}\tLoss={:.4f}\tTop1 Accuracy={:.4f}\tTop5 Accuracy={:.4f}'
                    .format(epoch, args.end_epoch, i, len(train_loader), get_learning_rate(optimizer), losses.avg, top1_accs.avg, top5_accs.avg))
    return losses.avg, top1_accs.avg, top5_accs.avg


def main():
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
