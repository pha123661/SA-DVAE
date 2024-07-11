import argparse
import pickle as pkl

import numpy as np
from sklearn.linear_model import LogisticRegression


parser = argparse.ArgumentParser(description='View adaptive')
parser.add_argument('--ss', type=int, required=True, help="split size")
parser.add_argument('--st', type=str, required=True, help="split type")
parser.add_argument('--dataset_path', type=str,
                    required=True, help="dataset path")
parser.add_argument('--dataset', type=str, required=True, help="dataset name")
parser.add_argument('--wdir', type=str, required=True,
                    help="directory to save weights path")
parser.add_argument('--le', type=str, required=True,
                    help="language embedding model")
parser.add_argument('--ve', type=str, required=True,
                    help="visual embedding model")
parser.add_argument('--phase', type=str, required=True, help="train or val")
parser.add_argument('--num_classes', type=int, required=True,
                    help="number of classes")
parser.add_argument('--tm', type=str, required=True, help='text mode')
parser.add_argument('--th', type=int,
                    required=True, help='threshold')
parser.add_argument('--t', type=int,  required=True, help='temp')
args = parser.parse_args()


num_unseen_classes = args.ss
st = args.st
dataset = args.dataset
dataset_path = args.dataset_path
wdir = args.wdir
le = args.le
ve = args.ve
phase = args.phase
num_classes = args.num_classes
tm = args.tm
th_set = args.th
t_set = args.t

seed = 5
np.random.seed(seed)


def temp_scale(seen_features, T):  # softmax
    return np.array([np.exp(i)/np.sum(np.exp(i)) for i in (seen_features + 1e-12)/T])


# unseen cls output of "ztest.npy", which is the features of the unseen validation set from training set heldout
unseen_zs = np.load(
    f'{wdir}/{le}/{tm}/MSF_{num_unseen_classes}_r_unseen_zs.npy')
# unseen cls output of "val.npy", which is the features of the seen validation set from training set heldout
seen_zs = np.load(f'{wdir}/{le}/{tm}/MSF_{num_unseen_classes}_r_seen_zs.npy')
# seen cls output of "ztest.npy"
unseen_train = np.load(f'{dataset_path}/ztest_out.npy')
# seen cls output of "val.npy"
seen_train = np.load(f'{dataset_path}/val_out.npy')

seen_random_idx = np.random.choice(
    np.arange(seen_train.shape[0]),
    min(unseen_train.shape[0], seen_train.shape[0]),
    replace=False
)
seen_train = seen_train[seen_random_idx]
seen_zs = seen_zs[seen_random_idx]

best_model = None
best_acc = 0
best_thresh = 0
t_iter = [i for i in range(1, 10)] if t_set == 0 else [t_set]
for t in t_iter:
    fin_val_acc = 0
    fin_train_acc = 0
    prob_unseen_zs = unseen_zs
    prob_unseen_train = temp_scale(unseen_train, t)
    prob_seen_zs = seen_zs
    prob_seen_train = temp_scale(seen_train, t)

    feat_unseen_zs = -np.sort(-prob_unseen_zs, 1)[:, :num_unseen_classes]
    feat_unseen_train = -np.sort(-prob_unseen_train, 1)[:, :num_unseen_classes]
    feat_seen_zs = -np.sort(-prob_seen_zs, 1)[:, :num_unseen_classes]
    feat_seen_train = -np.sort(-prob_seen_train, 1)[:, :num_unseen_classes]
    val_unseen_inds = np.random.choice(np.arange(
        feat_unseen_train.shape[0]), min(300, feat_unseen_train.shape[0] // 6), replace=False)
    val_seen_inds = np.random.choice(np.arange(
        feat_seen_train.shape[0]), min(300, feat_seen_train.shape[0] // 6), replace=False)
    train_unseen_inds = np.setdiff1d(
        np.arange(feat_unseen_train.shape[0]), val_unseen_inds)
    train_seen_inds = np.setdiff1d(
        np.arange(feat_seen_train.shape[0]), val_seen_inds)

    gating_train_x = np.concatenate([
        np.concatenate([
            feat_unseen_zs[train_unseen_inds],
            feat_unseen_train[train_unseen_inds]], axis=1),
        np.concatenate([
            feat_seen_zs[train_seen_inds],
            feat_seen_train[train_seen_inds]], axis=1),], axis=0)
    gating_train_y = np.array(
        [0]*len(train_unseen_inds) + [1]*len(train_seen_inds))
    gating_val_x = np.concatenate([
        np.concatenate([
            feat_unseen_zs[val_unseen_inds],
            feat_unseen_train[val_unseen_inds]], 1),
        np.concatenate([
            feat_seen_zs[val_seen_inds],
            feat_seen_train[val_seen_inds]], 1)], 0)
    gating_val_y = np.array([0]*len(val_unseen_inds) + [1]*len(val_seen_inds))

    train_inds = np.arange(gating_train_x.shape[0])
    np.random.shuffle(train_inds)

    model = LogisticRegression(random_state=0, C=1, solver='lbfgs', n_jobs=2,
                               multi_class='multinomial', verbose=0, max_iter=50000,
                               ).fit(gating_train_x[train_inds], gating_train_y[train_inds])

    prob = model.predict_proba(gating_val_x)
    best = 0
    bestT = 0
    th_iter = [i for i in range(25, 75, 1)] if th_set == 0 else [
        th_set]  # (25, 75) -> (45, 55)
    for th in th_iter:
        y = prob[:, 0] > th/100
        acc = np.sum((1 - y) == gating_val_y)/len(gating_val_y)
        if acc > best:
            best = acc
            bestT = th/100
    fin_val_acc += best
    pred_train = model.predict(gating_train_x)
    train_acc = np.sum(pred_train == gating_train_y)/len(gating_train_y)
    fin_train_acc += train_acc
    print('gating_train_x', gating_train_x.shape, 'gating_train_y',
          gating_train_y.shape, 'gating_val_x', gating_val_x.shape, 'gating_val_y', gating_val_y.shape)
    # print first sample with y = 0 and y=  1 respectively
    print(gating_train_x[0], gating_train_y[0])
    print(gating_train_x[-1], gating_train_y[-1])

    if fin_val_acc > best_acc:
        best_temp = t
        best_acc = fin_val_acc
        best_thresh = bestT
        best_model = model

print('best validation accuracy for the gating model', best_acc)
print('best threshold', best_thresh)
print('best temperature', best_temp)

with open(wdir.replace('_val', '') + f'/{le}/{tm}/gating_model.pkl', 'wb') as num_unseen_classes:
    pkl.dump(best_model, num_unseen_classes)
    num_unseen_classes.close()
