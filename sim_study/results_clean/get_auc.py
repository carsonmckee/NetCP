import csv
import itertools
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

def compute_autnr(y_true, y_scores, num_thresholds=200):
    """
    Compute Area Under the True Negative Rate curve (AUTNR).

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True binary labels (0 or 1). Assumes mostly or all zeros.
    - y_scores: array-like of shape (n_samples,)
        Predicted scores or probabilities.
    - num_thresholds: int, optional (default=100)
        Number of thresholds to evaluate between 0 and 1.

    Returns:
    - autnr: float
        Area Under the True Negative Rate curve.
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    thresholds = np.linspace(0, 1, num_thresholds)
    tnr_values = []

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        tnr = TN / (TN + FP) if (TN + FP) > 0 else 0
        tnr_values.append(tnr)

    # plt.plot(thresholds, tnr_values)
    # plt.show()
    autnr = np.trapz(tnr_values, thresholds)
    return autnr

def compute_auc(true_labels, pred_scores):
    # Separate positive and negative examples
    positives = []
    negatives = []
    
    for label, score in zip(true_labels, pred_scores):
        if label == 1:
            positives.append(score)
        else:
            negatives.append(score)
    
    n_positive = len(positives)
    n_negative = len(negatives)

    if n_positive == 0 or n_negative == 0:
        raise ValueError("Both positive and negative samples are needed to compute AUC.")
    
    count = 0.0
    for pos_score in positives:
        for neg_score in negatives:
            if pos_score > neg_score:
                count += 1
            elif pos_score == neg_score:
                count += 0.5

    auc = count / (n_positive * n_negative)
    return auc

def argsort(seq, reverse=False):
    return sorted(range(len(seq)), key=lambda x: seq[x], reverse=reverse)

def flatten(a:list):
    flat = []
    for i in range(len(a)):
        for j in range(len(a)):
            if i != j:
                flat.append(a[i][j])
    return flat

def quantile(data, q):
    if not 0 <= q <= 1:
        raise ValueError("q should be between 0 and 1")

    data_sorted = sorted(data)
    n = len(data_sorted)

    # Position (using linear interpolation between closest ranks if necessary)
    pos = q * (n - 1)
    lower_index = int(pos)
    upper_index = lower_index + 1
    weight = pos - lower_index

    if upper_index >= n:
        return data_sorted[lower_index]
    else:
        return (1 - weight) * data_sorted[lower_index] + weight * data_sorted[upper_index]

def mean(a:list):
    return sum(a) / len(a)

if __name__ == "__main__":
    
    true_S1 = flatten([[1, 1, 0, 0], 
                        [0, 1, 1, 0], 
                        [0, 0, 1, 1], 
                        [0, 0, 0, 1]])
    true_S2 = flatten([[1, 1, 0, 0], 
                        [0, 1, 0, 0], 
                        [0, 0, 1, 1], 
                        [0, 0, 0, 1]])
    true_S5 = flatten([[1, 0, 0, 0], 
                        [0, 1, 0, 0], 
                        [0, 0, 1, 0], 
                        [0, 0, 0, 1]])
    a = 5 / 100
    d = 3
    for likelihood in ['normal_mean', 'ar_process']:
        for scenario in [1, 2]:
            aucs = []
            aps = []
            for i in range(1, 51, 1):
                path = f"{likelihood}/adj/NetCP_{scenario}_{i}.csv"
                mat = []
                with open(path, 'r') as file:
                    reader = csv.reader(file)
                    for row in reader:
                        mat.append([float(i) for i in row])
                mat = flatten(mat)
                if scenario == 1:
                    aucs.append(roc_auc_score(true_S1, mat))
                    aps.append(average_precision_score(true_S1, mat))
                elif scenario == 2:
                    aucs.append(roc_auc_score(true_S2, mat))
                    aps.append(average_precision_score(true_S1, mat))
                else:
                    aucs.append(compute_autnr(true_S5, mat))
                    # aucs.append(compute_auc(true_S5, mat))
            
            # if (likelihood == 'normal_mean') and (scenario == 2):
            #     print(np.argmin(aucs))
            # print(sorted(aucs))
            print(f"AUC for {likelihood} - S{scenario} = {round(np.mean(aucs), d)} ({round(np.quantile(aucs, a/2), d)}, {round(np.quantile(aucs, 1-a/2), d)})")
            # print(f"PR-AUC for {likelihood} - S{scenario} = {round(np.mean(aps), d)} ({round(np.quantile(aps, a/2), d)}, {round(np.quantile(aps, 1-a/2), d)})")
            
        
    
    # for scenario 5 report the average TNR with quantiles when thresholding at 50%
    thresh = 0.5
    for likelihood in ['normal_mean', 'ar_process']:
        TNRs = []
        
        for i in range(1, 51, 1):
            path = f"{likelihood}/adj/NetCP_5_{i}.csv"
            mat = []
            with open(path, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    mat.append([float(i) for i in row])
            mat = flatten(mat)
            TNR = sum(1 if pred < thresh else 0 for pred in mat) / len(mat)
            TNRs.append(TNR)
        
        print(f"TNR (threshold = {thresh}) for {likelihood} - S5 = {round(np.mean(TNRs), d)} ({round(np.quantile(TNRs, a/2), d)}, {round(np.quantile(TNRs, 1-a/2), d)})")