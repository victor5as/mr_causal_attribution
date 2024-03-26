"""
This code implements additional simulations for the following paper:
Multiply-Robust Causal Change Attribution (2024)
Anonymous Authours
"""

import numpy as np
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from scipy.stats import sem
from mr_attribution import ThetaC
from joblib import Parallel, delayed, dump
import pathlib

def run_sim(seed=0, K=30, N=2000, m=0.2, r=0.5):    
    
    # 1. Generate Data
    np.random.seed(seed)
    change_ind = np.random.choice(range(K+1), K//10, replace=False)
    data0 = np.zeros((N, K+1))
    data0[:,0] = np.random.normal(0,1,N)
    for k in range(1, K+1):
        data0[:,k] = np.random.normal(r*data0[:,k-1], np.sqrt(1.0-r**2),N)
    data1 = np.zeros((N, K+1))
    data1[:,0] = np.random.normal(0,1,N) if 0 not in change_ind else np.random.normal(1,1,N)
    for k in range(1, K+1):
        data1[:,k] = (np.random.normal(r*data1[:,k-1], np.sqrt(1.0-r**2), N) if k not in change_ind 
                      else np.random.normal(r*data1[:,k-1]+m, np.sqrt(1.0-r**2), N))
    
    X, y = np.concatenate((data0[:,:-1], data1[:,:-1])), np.concatenate((data0[:,-1], data1[:,-1]))
    T = np.concatenate((np.zeros(N), np.ones(N)))

    X_train, X_eval, y_train, y_eval, T_train, T_eval = train_test_split(X, y, T, train_size = 0.8, 
                                                                         stratify = T, random_state = seed)

    # 2. True Path Attribution Measure
    betas = np.array([r**(K-k) for k in range(K+1)])
    path_true = np.array([(k in change_ind)*betas[k]*m for k in range(K+1)])
    
    # 3. Estimate Path Attribution Measures
    theta_est = {}
    path_est = {}
    MAE = {}
    
    for method in ['regression', 're-weighting', 'MR']:
        theta_est[method] = np.zeros(K+2)
        theta_est[method][0] = np.mean(data0[:,-1])
        theta_est[method][-1] = np.mean(data1[:,-1])
    
        for k in range(1, K+1):
            C = np.array([1 for _ in range(k)] + [0 for _ in range(k, K+1)])
            theta_est[method][k], _ = ThetaC(C, warn_thr=-1).est_theta(X_train, y_train, T_train, 
                                                                       X_eval, y_eval, T_eval,
                                                                       method=method,
                                                                       regressor=LassoCV,
                                                                       classifier=LogisticRegressionCV,
                                                                       classifier_kwargs={"penalty" : "l1", "solver" : "liblinear"})
    
        path_est[method] = np.array([theta_est[method][k+1] - theta_est[method][k] for k in range(K+1)])
        MAE[method] = np.abs(path_est[method] - path_true)

    return MAE

def main():
    pathlib.Path('results/').mkdir(exist_ok=True) 

    for K in [10, 20, 50, 100]:
        res = Parallel(n_jobs = 25, verbose = 3)(delayed(run_sim)(seed=seed, K=K) for seed in range(500))
        
        res_collapsed = {}
        for k in ['regression', 're-weighting', 'MR']:
            res_collapsed[k] = np.vstack([x[k] for x in res])
        dump(res_collapsed, f"results/K_{K}.joblib")
        

        print(f"K = {K}, worst-case MAE:")
        for k, x in res_collapsed.items(): 
            print(f"  {k} = {x.max(axis=1).mean():.3f} +/- {sem(x.max(axis=1)):.3f}")  
        print("")

if __name__ == "__main__":
    main()
