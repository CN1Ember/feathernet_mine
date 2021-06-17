import torch
import numpy as np
from sklearn.metrics import roc_curve, auc

# def count_scores(embeddings1, embeddings2):
#     score = np.zeros((len(embeddings1),))
#     for i in range(len(embeddings1)):
#         feat1 = embeddings1[i]
#         feat2 = embeddings2[i]
#         similarity_score = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * (np.linalg.norm(feat2)))
#         score[i] = similarity_score
#     return score

def count_scores(embeddings1, embeddings2):
    return np.sum(embeddings1 * embeddings2, -1)

def count_roc(issame, scores, fpr):
    #x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
    x_labels = [fpr]
    results = []
    fpr, tpr, _ = roc_curve(issame, scores)
    #roc_auc = auc(fpr, tpr)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr) # select largest tpr at same fpr
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(list(zip(abs(fpr-x_labels[fpr_iter]), range(len(fpr)))))
        results.append('%.4f' % tpr[min_index])
    return results


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

def val_verification(args, model, carray, issame, fpr=10**-6):
    model.eval()
    # 提取特征
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

    # 计算cos similarly
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    scores = count_scores(embeddings1, embeddings2)

    # count roc
    results = count_roc(issame, scores, fpr)
    return float(results[0])


def val_verification_test(model, carray, issame, fpr=10**-6, emb_size=512, batch_size=128):
    model.eval()
    # 提取特征
    idx = 0
    embeddings = np.zeros([len(carray), emb_size])
    with torch.no_grad():
        while idx + batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx + batch_size])
            embeddings[idx:idx + batch_size] = l2_norm(model(batch.cuda()).cpu())  # xz: add l2_norm
            idx += batch_size
        if idx < len(carray):
            batch = torch.tensor(carray[idx:])
            embeddings[idx:] = l2_norm(model(batch.cuda()).cpu())   # xz: add l2_norm

    # 计算cos similarly
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    scores = count_scores(embeddings1, embeddings2)

    # count roc
    results = count_roc(issame, scores, fpr)
    return float(results[0])

