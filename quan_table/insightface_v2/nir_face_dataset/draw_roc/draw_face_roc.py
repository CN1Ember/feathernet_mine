import os, torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing


# feature_size = 128
feature_size = 512

def count_scores(embeddings1, embeddings2):
    return np.sum(embeddings1 * embeddings2, -1)


def read_feat(f_path):
    # print(f_path)
    f_path = f_path.replace("\\", "/")
    # print(f_path)
    feat = np.loadtxt(f_path, dtype=np.float32).reshape(1, feature_size)
    # print(feat.shape)
    feat = sklearn.preprocessing.normalize(feat)
    return feat

# 画ROC曲线图
def draw_roc_curve(fpr, tpr):
    plt.figure()
    plt.semilogx(fpr, tpr, label='r34')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Face Verification ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(linestyle='--', linewidth=1)
    plt.show()
    # plt.savefig(save_name+'.pdf')


# windows
# root = '/home/yezilong/jidian_face_112x112_499_align_clear_mxnet_resnet34_gpu'
root = 'D:/2019Programs/jidian2019/极点智能/Draw ROC/jidian_face_112x112_499_align_result/' \
       'jidian_face_112x112_499_align_clear_mxnet_resnet34_gpu'

# root = 'D:/2019Programs\jidian2019\极点智能\Draw ROC\jidian_face_112x112_499_align_result\mobilefacenet_pruned' \
#        '\mobilefacenet_cos_lr_pruned_0.5_nowith_fc_checkpoint_026'

# linux
# root = '/home/dataset/xz_datasets/jidian_face_499/feature_result/jidian_face_112x112_499_align_clear_mxnet_resnet34_gpu'


# 收集arr (图片名 label)
arr = []
txt_path = './list.txt'
f = open(txt_path, 'w+')
root = root.replace("\\", "/")
for d in os.listdir(root):
    d_path =  os.path.join(root, d)
    if os.path.isdir(d_path):
        for img in os.listdir(d_path):
            arr.append((img, d))
            # f.write(os.path.join(root, d, img)+'\n')
            print(os.path.join(root, d, img))

# f.close()
all_feat = []
for i in range(len(arr)):
    feat = read_feat(os.path.join(root, arr[i][1], arr[i][0]))  # root+label+img_name.txt
    all_feat.append(feat)


# count roc
issame = []
scores = []
pos_pair = 0
neg_pair= 0
for i in range(len(arr)):
    feat1 = all_feat[i]
    # print('feat1.shape={}'.format(feat1.shape))
    for j in range(i + 1, len(arr)):
        feat2 = all_feat[j]
        # print('feat2.shape={}'.format(feat2.shape))
        similarity_score = count_scores(feat1, feat2)
        # print('similarity_score.shape={}'.format(similarity_score.shape))
        scores.append(similarity_score)

        if arr[i][1] == arr[j][1]:
            issame.append(1)
            pos_pair += 1
            f.write('{}, {}\n'.format(arr[i][1]+'/'+arr[i][0], arr[j][1]+'/'+arr[j][0]))
        else:
            issame.append(0)
            neg_pair += 1

    # print(len(issame))
    # print(len(scores), scores[0].shape)
    # assert False
    print(i, len(arr))
print('pos_pair={}'.format(pos_pair))
print('neg_pair={}'.format(neg_pair))
print('total_pair={}'.format(len(issame)))
# break


print('count roc')
# count_roc(issame, scores)
fpr, tpr, threshold1s = roc_curve(issame, scores)

np.save('./r34_roc', (fpr, tpr, threshold1s))
draw_roc_curve(fpr, tpr)