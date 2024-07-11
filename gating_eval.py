import argparse
import pickle as pkl

import numpy as np
import torch


parser = argparse.ArgumentParser(description='View adaptive')
parser.add_argument('--ss', type=int, help="split size", required=True)
parser.add_argument('--st', type=str, help="split type", required=True)
parser.add_argument('--dataset_path', type=str,
                    help="dataset path", required=True)
parser.add_argument('--dataset', type=str, help="dataset name", required=True)
parser.add_argument('--wdir', type=str,
                    help="directory to save weights path", required=True)
parser.add_argument(
    '--le', type=str, help="language embedding model", required=True)
parser.add_argument(
    '--ve', type=str, help="visual embedding model", required=True)
parser.add_argument('--phase', type=str, help="train or val", required=True)
parser.add_argument('--temp', type=int, help="temperature", required=True)
parser.add_argument('--num_classes', type=int,
                    help="num_classes", required=True)
parser.add_argument('--thresh', type=float, help="temperature", required=True)
parser.add_argument('--tm', type=str, help='text mode', required=True)

args = parser.parse_args()

ss = args.ss
st = args.st
dataset = args.dataset
dataset_path = args.dataset_path
wdir = args.wdir
le = args.le
ve = args.ve
phase = args.phase
num_classes = args.num_classes
temp = args.temp
thresh = args.thresh
tm = args.tm

seed = 5
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

if phase == 'val':
    gzsl_inds = np.load(
        f'resources/label_splits/{dataset}/{st}s{num_classes - ss}.npy')
    unseen_inds = np.sort(
        np.load(f'resources/label_splits/{dataset}/{st}v{ss}_0.npy'))
    seen_inds = np.load(
        f'resources/label_splits/{dataset}/{st}s{num_classes - ss - ss}_0.npy')
else:
    gzsl_inds = np.arange(num_classes)
    unseen_inds = np.sort(
        np.load(f'resources/label_splits/{dataset}/{st}u{ss}.npy'))
    seen_inds = np.load(
        f'resources/label_splits/{dataset}/{st}s{num_classes - ss}.npy')

tars = np.load(dataset_path + '/g_label.npy')

test_y = []
for i in tars:
    if i in unseen_inds:
        test_y.append(0)
    else:
        test_y.append(1)

test_zs = np.load(f'{wdir}/{le}/{tm}/MSF_{str(ss)}_r_gzsl_zs.npy')
test_seen = np.load(dataset_path + '/gtest_out.npy')


def temp_scale(seen_features, T):
    return np.array([np.exp(i)/np.sum(np.exp(i)) for i in (seen_features + 1e-12)/T])


prob_test_zs = test_zs
prob_test_seen = temp_scale(test_seen, temp)

feat_test_zs = np.sort(prob_test_zs, 1)[:, ::-1][:, :ss]
feat_test_seen = np.sort(prob_test_seen, 1)[:, ::-1][:, :ss]

gating_test_x = np.concatenate([feat_test_zs, feat_test_seen], 1)
gating_test_y = test_y


with open(f'{wdir}/{le}/{tm}/gating_model.pkl', 'rb') as f:
    gating_model = pkl.load(f)

prob_gate = gating_model.predict_proba(gating_test_x)
pred_test = 1 - prob_gate[:, 0] > thresh
np.sum(pred_test == test_y)/len(test_y)
a = prob_gate
b = np.zeros(prob_gate.shape[0])
p_gate_seen = prob_gate[:, 1]
prob_y_given_seen = prob_test_seen + \
    (1/num_classes)*np.repeat((1 - p_gate_seen)[:, np.newaxis], num_classes, 1)
p_gate_unseen = prob_gate[:, 0]
prob_y_given_unseen = prob_test_zs + \
    (1/ss)*np.repeat((1 - p_gate_unseen)[:, np.newaxis], ss, 1)
prob_seen = prob_y_given_seen * \
    np.repeat(p_gate_seen[:, np.newaxis], num_classes, 1)
prob_unseen = prob_y_given_unseen * \
    np.repeat(p_gate_unseen[:, np.newaxis], ss, 1)

final_preds = []
seen_count = 0
tot_seen = 0
unseen_count = 0
tot_unseen = 0
gseen_count = 0
gunseen_count = 0
for i in range(len(gating_test_y)):
    if pred_test[i] == 1:
        pred = seen_inds[np.argmax(prob_test_seen[i, seen_inds])]
    else:
        pred = unseen_inds[np.argmax(prob_test_zs[i, :])]

    if tars[i] in seen_inds:
        tot_seen += 1
        if pred_test[i] == 1:
            gseen_count += 1
        if pred == tars[i]:
            seen_count += 1
    else:
        if pred_test[i] == 0:
            gunseen_count += 1
        tot_unseen += 1
        if pred == tars[i]:
            unseen_count += 1
    final_preds.append(pred)

seen_acc = seen_count/tot_seen
print(f'seen_accuracy: {seen_acc :.2%}')
unseen_acc = unseen_count/tot_unseen
print(f'unseen_accuracy: {unseen_acc :.2%}')
h_mean = 2*seen_acc*unseen_acc/(seen_acc + unseen_acc)
print(f'h_mean: {h_mean :.2%}')
