## script to compute mem and infl

import os
import pickle
import numpy as np
import pdb

def mem(mask, acc):
    inv_mask = np.logical_not(mask)
    u = (acc * mask).sum(axis = 0) / mask.sum(axis = 0)
    l = (acc * inv_mask).sum(axis = 0) / inv_mask.sum(axis = 0)

    return u - l

def infl(mask, acc):
    inv_mask = np.logical_not(mask)
    ## careful that mask & acc are both boolean! force the output be boolean!!
    M = mask / mask.sum(axis = 0)[None, :] ## M will be float number
    u = M.T @ acc
    inv_M = inv_mask / inv_mask.sum(axis = 0)[None, :] 
    l = inv_M.T @ acc

    return u - l



dir='../out'         # Replace with path to your directory: absolute or relative
pattern = 'pkl' # Replace with your target substring
matching_files = [f for f in os.listdir(dir) if pattern in f]

n_exp = len(matching_files)
n_train = 50000
n_test = 10000
n_test_new = 2000

mask = np.zeros((n_exp, n_train), bool)
acc_train = np.empty((n_exp, n_train), bool)
acc_test = np.empty((n_exp, n_test), bool)
acc_test_new = np.empty((n_exp, n_test_new), bool)


for i in range(n_exp):
    with open(os.path.join(dir, matching_files[i]), "rb") as f:
        data = pickle.load(f)
    mask[i, data["indices"]] = True
    acc_train[i, :] = data["acc_train"]
    acc_test[i, :] = data["acc_test"]
    acc_test_new[i, :] = data["acc_test_new"]

mem_score = mem(mask, acc_train)
idx = np.where(mem_score > np.quantile(mem_score, 0.99))[0] ## index for samples with biggest mem

infl_score_test = infl(mask[:, idx], acc_test)
infl_score_testnew = infl(mask[:, idx], acc_test_new)

out = {}
out["mask"] = mask
out["acc_train"] = acc_train
out["acc_test"] = acc_test
out["acc_test_new"] = acc_test_new
out["mem_score"] = mem_score
out["idx"] = idx
out["infl_score_test"] = infl_score_test
out["infl_score_testnew"] = infl_score_testnew

with open("../out/cifar10_summary.pkl", "wb") as f:
    pickle.dump(out, f)



