import os, torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing


# 画ROC曲线图
def draw_roc_curve(fpr, tpr):
    plt.figure()
    plt.semilogx(fpr, tpr, label='r34')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Face Verification ROC Curve')
    plt.yticks(np.linspace(0.0, 1.0, 11, endpoint=True))
    plt.legend(loc="lower right")
    plt.grid(linestyle='--', linewidth=1)
    plt.show()
    plt.savefig('roc.pdf')


fpr, tpr, thr = np.load('./r34_roc.npy')

# for i in range(len(fpr)):
#     print(fpr[i])
print('\n')
# for i in range(len(tpr)):
#     print(tpr[i])

# print('FAR=1, tpr={}, thr={}'.format(tpr[-1], thr[-1]))
# assert False

draw_roc_curve(fpr,tpr)
