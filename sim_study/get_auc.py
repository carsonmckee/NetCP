import csv
import numpy as np
from sklearn.metrics import roc_auc_score

def flatten(a:list):
    flat = []
    for i in range(len(a)):
        for j in range(len(a)):
            if i != j:
                flat.append(a[i][j])
    return flat

if __name__ == "__main__":
    
    true_S1 = flatten([[1, 1, 0, 0], 
                        [0, 1, 1, 0], 
                        [0, 0, 1, 1], 
                        [0, 0, 0, 1]])
    true_S2 = flatten([[1, 1, 0, 0], 
                        [0, 1, 0, 0], 
                        [0, 0, 1, 1], 
                        [0, 0, 0, 1]])

    alpha = 5 / 100
    decimals = 3
    for likelihood in ['normal_mean', 'ar_process']:
        for scenario in [1, 2]:
            aucs = []
            for i in range(1, 51, 1):
                path = f"results_clean/{likelihood}/adj/NetCP_{scenario}_{i}.csv"
                mat = []
                with open(path, 'r') as file:
                    reader = csv.reader(file)
                    for row in reader:
                        mat.append([float(i) for i in row])
                mat = flatten(mat)
                if scenario == 1:
                    aucs.append(roc_auc_score(true_S1, mat))
                elif scenario == 2:
                    aucs.append(roc_auc_score(true_S2, mat))
            
            print(f"AUC for {likelihood} - S{scenario} = {round(np.mean(aucs), decimals)} ({round(np.quantile(aucs, alpha/2), decimals)}, {round(np.quantile(aucs, 1-alpha/2), decimals)})")
            
        
    
    # for scenario 5 report the average TNR with quantiles when thresholding at 50%
    thresh = 0.5
    for likelihood in ['normal_mean', 'ar_process']:
        TNRs = []
        
        for i in range(1, 51, 1):
            path = f"results_clean/{likelihood}/adj/NetCP_5_{i}.csv"
            mat = []
            with open(path, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    mat.append([float(i) for i in row])
            mat = flatten(mat)
            TNR = sum(1 if pred < thresh else 0 for pred in mat) / len(mat)
            TNRs.append(TNR)
        
        print(f"TNR (threshold = {thresh}) for {likelihood} - S5 = {round(np.mean(TNRs), decimals)} ({round(np.quantile(TNRs, alpha/2), decimals)}, {round(np.quantile(TNRs, 1-alpha/2), decimals)})")