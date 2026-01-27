import numpy as np
from munkres import Munkres, print_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment as linear
from sklearn import metrics


def cluster_acc(y_true, y_pred):
    """
    Clustering accuracy via optimal assignment (Hungarian).

    Unlike the original implementation, this is robust to a mismatch between the number of
    unique labels in y_true vs y_pred (e.g., collapse to fewer clusters).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0, 0.0
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"y_true and y_pred size mismatch: {y_true.shape} vs {y_pred.shape}")

    # Relabel to contiguous ids starting at 0 for a compact confusion matrix.
    _, y_true_ids = np.unique(y_true, return_inverse=True)
    _, y_pred_ids = np.unique(y_pred, return_inverse=True)

    n_true = int(y_true_ids.max()) + 1
    n_pred = int(y_pred_ids.max()) + 1
    dim = int(max(n_true, n_pred))

    w = np.zeros((dim, dim), dtype=np.int64)
    for i in range(y_true_ids.size):
        w[y_pred_ids[i], y_true_ids[i]] += 1

    # Maximize agreement => minimize (max - w).
    row_ind, col_ind = linear(w.max() - w)
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind) if int(r) < n_pred and int(c) < n_true}

    mapped = np.zeros_like(y_pred_ids)
    for r in range(n_pred):
        mapped[y_pred_ids == r] = mapping.get(r, 0)

    acc = metrics.accuracy_score(y_true_ids, mapped)
    f1_macro = metrics.f1_score(y_true_ids, mapped, average='macro')
    return float(acc), float(f1_macro)


def eva(y_true, y_pred, epoch=0):
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    print(epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
            ', f1 {:.4f}'.format(f1))

    # 返回评估结果
    return acc, f1, nmi, ari
