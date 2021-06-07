import numpy as np

from scipy import interpolate

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def cal_metric(target, predicted,show = False, save = False):
    fpr, tpr, thresholds = roc_curve(target, predicted)
    roc_result=open("./results/roc_test_nofmap_nir_0313_best.txt",'w')
    # roc_result.write("thresholds \t fpr \t tpr \n")
    for i in range(fpr.shape[0]):
        roc_result.write(str(thresholds[i])+ ' ' + str(fpr[i])+' '+str(tpr[i])+'\n')
    roc_result.close()
    _tpr = (tpr)
    _fpr = (fpr)
    tpr = tpr.reshape((tpr.shape[0],1))
    fpr = fpr.reshape((fpr.shape[0],1))
    scale = np.arange(0, 1, 0.00000001)
    function = interpolate.interp1d(_fpr, _tpr)
    y = function(scale)
    znew = abs(scale + y -1)
    eer = scale[np.argmin(znew)]
    FPRs = {"TPR@FPR=10E-2": 0.01, "TPR@FPR=10E-3": 0.001, "TPR@FPR=10E-4": 0.0001}
    TPRs = {"TPR@FPR=10E-2": 0.01, "TPR@FPR=10E-3": 0.001, "TPR@FPR=10E-4": 0.0001}
    for i, (key, value) in enumerate(FPRs.items()):
        index = np.argwhere(scale == value)
        score = y[index] 
        TPRs[key] = float(np.squeeze(score))
        print(score)

    auc = roc_auc_score(target, predicted)
    if show:
        plt.plot(scale, y)
        plt.show()
    if save:
        plt.plot(scale, y)
        plt.title('ROC')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.savefig('./results/roc_1.0_93_best.png')
    return eer,TPRs, auc,{'x':scale, 'y':y}

if __name__ == "__main__":
   f1 = open('/home/lidaiyuan/feathernet2020/FeatherNet/submission/exp_train_set_21030602_exp_20210312173024_2021-03-15_15:22:19_FaceFeatherNet_v3_0_submission_gt.txt','r')
   f2 = open('/home/lidaiyuan/feathernet2020/FeatherNet/submission/exp_train_set_21030602_exp_20210312173024_2021-03-15_15:22:19_FaceFeatherNet_v3_0_submission.txt','r')
   label = [int(i) for i in f1.read().splitlines()]
   pre = [float(i) for i in f2.read().splitlines()]
   cal_metric(label,pre,False,True)

