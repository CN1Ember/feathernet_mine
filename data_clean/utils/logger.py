import logging
import os
import sys
import torch
import numpy as np
from torch import nn
from utils.verifacation import evaluate
from utils.utils import get_learning_rate
from utils.model_analyse import ModelAnalyse


def get_logger(save_path, logger_name):
    """
    Initialize logger
    """

    logger = logging.getLogger(logger_name)
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    # file log
    file_handler = logging.FileHandler(os.path.join(save_path, "experiment.log"))
    file_handler.setFormatter(file_formatter)

    # console log
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.setLevel(logging.INFO)
    return logger


def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)
    return hfliped_imgs


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def board_val(args,logger, writer, db_name, accuracy, best_threshold, step):
    logger.info('||===>>Test Epoch: [[{:d}/{:d}]\t\tVal:{}_accuracy={:.4f}%\t\tbest_threshold={:.4f}'
                .format(step, args.end_epoch, db_name, accuracy*100, best_threshold))
    writer.add_scalar('val_{}_accuracy'.format(db_name), accuracy*100, step)
    writer.add_scalar('val_{}_best_threshold'.format(db_name), best_threshold, step)
    # writer.add_image('val_{}_roc_curve'.format(db_name), roc_curve_tensor, step)


def val_evaluate(args, model, carray, issame, nrof_folds = 10, use_flip=True):
    model.eval()
    idx = 0
    embeddings = np.zeros([len(carray), args.emb_size])
    with torch.no_grad():
        while idx + args.batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx + args.batch_size])
            out = model(batch.cuda())
            if use_flip:
                fliped_batch = hflip_batch(batch)
                fliped_out = model(fliped_batch.cuda())
                out = out + fliped_out

            embeddings[idx:idx + args.batch_size] = l2_norm(out).cpu()  # xz: add l2_norm
            idx += args.batch_size
        if idx < len(carray):
            batch = torch.tensor(carray[idx:])
            out = model(batch.cuda())

            if use_flip:
                fliped_batch = hflip_batch(batch)
                fliped_out = model(fliped_batch.cuda())
                out = out + fliped_out

            embeddings[idx:] = l2_norm(out).cpu()  # xz: add l2_norm
    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    # buf = gen_plot(fpr, tpr)
    # roc_curve = Image.open(buf)
    # roc_curve_tensor = transforms.ToTensor()(roc_curve)
    return accuracy.mean(), best_thresholds.mean()


def logger_model_size(logger, model):
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


def logger_record_loss_acc(writer, model, logger, params, dataset_params, best_acc_params):
    [args, train_loss, epoch, optimizer, train_top1_accs, train_top5_accs] = params
    [agedb_30, agedb_30_issame, lfw, lfw_issame, cfp_fp, cfp_fp_issame] = dataset_params
    [tpr_best, lfw_best_acc, agedb_best_acc, cfp_best_acc] = best_acc_params
    writer.add_scalar('Train_Loss', train_loss, epoch)
    writer.add_scalar('Train_lr', get_learning_rate(optimizer), epoch)
    writer.add_scalar('Train_Top1_Accuracy', train_top1_accs, epoch)
    writer.add_scalar('Train_Top5_Accuracy', train_top5_accs, epoch)
    # One epoch's validation
    agedb_accuracy, agedb_best_threshold = val_evaluate(args, model, agedb_30, agedb_30_issame)
    board_val(args, logger, writer, 'agedb_30', agedb_accuracy, agedb_best_threshold, epoch)
    lfw_accuracy, lfw_best_threshold = val_evaluate(args, model, lfw, lfw_issame)
    board_val(args, logger, writer, 'lfw', lfw_accuracy, lfw_best_threshold, epoch)
    cfp_accuracy, cfp_best_threshold = val_evaluate(args, model, cfp_fp, cfp_fp_issame)
    board_val(args, logger, writer, 'cfp_fp', cfp_accuracy, cfp_best_threshold, epoch)
    tpr_best = max(lfw_accuracy, tpr_best)
    lfw_best_acc = max(lfw_accuracy, lfw_best_acc)
    agedb_best_acc = max(agedb_accuracy, agedb_best_acc)
    cfp_best_acc = max(cfp_accuracy, cfp_best_acc)
    return tpr_best, lfw_best_acc, agedb_best_acc, cfp_best_acc

