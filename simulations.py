"""
This module replicates the simulations in the following paper:
Multiply-Robust Causal Change Attribution (2024)
Anonymous Authours
"""

import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import  PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import sem
from math import comb
from joblib import Parallel, delayed, dump
from mr_attribution import ThetaC
import itertools
import matplotlib.pyplot as plt
import pathlib

# Number of explanatory variables
K = 2

# Get all possible 3- and 2-vectors of {True, False}
l = [False, True]
all_combos = [np.array(i) for i in itertools.product(l, repeat=K+1)]
all_combos_minus1 = [np.array(i) for i in itertools.product(l, repeat=K)]

# Function to generate synthetic data (same distributions as in the paper)
def gen_data(seed=0, N=1000):    
    np.random.seed(seed)
    X0 = np.random.normal(1,1,N)
    X0 = np.c_[X0, 0.5*X0 + np.random.normal(0,1,N)]
    Y0 = X0[:,0] + X0[:,1] + 0.25*X0[:,0]**2 + 0.25*X0[:,1]**2 + np.random.normal(0,1,N)
    X1 = np.random.normal(1,1.1,N)
    X1 = np.c_[X1, 0.2*X1 + np.random.normal(0,1,N)]
    Y1 = X1[:,0] + X1[:,1] + 0.25*X1[:,0]**2 - 0.25*X1[:,1]**2 + np.random.normal(0,1,N)
    X = np.concatenate((X0, X1))
    y = np.concatenate((Y0, Y1))
    T = np.concatenate((np.zeros(N), np.ones(N)))
    return X, y, T

# True values of each theta^C
def true_theta(C):
    mX0, mX0sq = (1,2.21) if C[0] else (1,2)
    mX1, mX1sq = (0.2*mX0,0.2**2*mX0sq+1) if C[1] else (0.5*mX0,0.5**2*mX0sq+1)
    mY = mX0 + mX1 + 0.25*mX0sq - 0.25*mX1sq if C[2] else mX0 + mX1 + 0.25*mX0sq + 0.25*mX1sq
    return mY

true_thetas = {}

for C in all_combos:
    true_thetas[''.join(map(lambda x : str(int(x)), C))] = true_theta(C) # The key will be 000, 001, 010, etc.

def est_all_thetas(X, y, T, method='MR', kwargs=None):
    kwargs = {} if kwargs is None else kwargs
    res = {}
    for C in all_combos:
        t, _ = ThetaC(C, warn_thr=-1).est_theta(X, y, T, X, y, T, method=method, **kwargs)
        res[''.join(map(lambda x : str(int(x)), C))] = t
    return res

def get_shap_val(res_dict):
    shap = np.zeros(K+1)

    for k in range(K+1):
        sv = 0.0

        for C in all_combos_minus1:
            C1 = np.insert(C, k, True)
            C0 = np.insert(C, k, False)
            chg = res_dict[''.join(map(lambda x : str(int(x)), C1))] - res_dict[''.join(map(lambda x : str(int(x)), C0))]
            sv += chg/((K+1)*comb(K, np.sum(C)))
        
        shap[k] = sv
    
    return shap

def run_sim(seed, spec, method="MR"):
    X, y, T = gen_data(seed)

    est_thetas = est_all_thetas(X, y, T, method=method, kwargs=spec['kwargs'])
    est_thetas_array = np.array([est_thetas[k] for k in true_thetas.keys()])
    est_shap = get_shap_val(est_thetas)
    
    return {'thetas' : est_thetas_array, 'shap' : est_shap}
        
# Get the true Shapley values
true_shap = get_shap_val(true_thetas)

specs = [
    # Correctly specified (Table 1a) 
    {'kwargs' : {'regressor' : [PolynomialFeatures, LinearRegression],
     'regressor_kwargs' : [{'degree' : 2}, {}], 'regressor_args' : [(), ()],
     'is_pipeline_reg' : True,
     'classifier' : [PolynomialFeatures, LogisticRegression], 'classifier_args' : [(), ()],
     'classifier_kwargs' : [{'degree' : 2}, {}],
     'is_pipeline_cla' : True},
     'fbasename'  : "results/param_"
    },
    
    # Incorrectly specified weights (Table 1b)
    {'kwargs' : {'regressor' : [PolynomialFeatures, LinearRegression],
     'regressor_kwargs' : [{'degree' : 2}, {}], 'regressor_args' : [(), ()],
     'is_pipeline_reg' : True,
     'classifier' : LogisticRegression,
     'classifier_kwargs' : {},},
     'fbasename'  : "results/weight_inc_"
    },
    
    # Incorrectly specified regression (Table 1c)
    {'kwargs' : {'regressor' : LinearRegression,
     'classifier' : [PolynomialFeatures, LogisticRegression], 'classifier_args' : [(), ()],
     'classifier_kwargs' : [{'degree' : 2}, {}],
     'is_pipeline_cla' : True},
     'fbasename'  : "results/regr_inc_"
    },

    # Both incorrectly specified (Table 1d)
    {'kwargs' : {'regressor' : LinearRegression,
     'classifier' : LogisticRegression},
     'fbasename'  : "results/both_inc_"
    },

    # Linear/Naive Bayes (not reported in the paper)
    {'kwargs' : {'regressor' : LinearRegression,
     'classifier' : GaussianNB,
     'classifier_kwargs' : {}},
     'fbasename'  : "results/linear_nb_"
    },

    # NN MLP (Table 1e):
    {'kwargs' : {'regressor'  : MLPRegressor,
     'regressor_kwargs' : {'hidden_layer_sizes' : (100,100,100), 'early_stopping' : True, 'random_state' : 0},
     'classifier' : CalibratedClassifierCV,
     'classifier_args' : (MLPClassifier(hidden_layer_sizes=(100,100,100), early_stopping=True, random_state=0),), 
     'classifier_kwargs' : {'cv' : 5}},
     'fbasename'  : "results/nn_"
    },

    # G-boost (Table 1f):
    {'kwargs' : {'regressor'  : GradientBoostingRegressor,
     'regressor_kwargs' : {'random_state' : 0},
     'classifier' : CalibratedClassifierCV,
     'classifier_args' : (GradientBoostingClassifier(random_state=0),), 
     'classifier_kwargs' : {'cv' : 5}},
     'fbasename'  : "results/gboost_"
    },
]

methods = ['regression', 're-weighting', 'MR']

def run_MC(spec, Nsim = 1000, print_MAE = True, print_bias = False, save = True):
    
    fbasename = spec['fbasename']

    true_thetas_array = np.array([true_thetas[k] for k in true_thetas.keys()])
    res_thetas = {'truth' : true_thetas_array}
    res_shap = {'truth' : true_shap}

    print(f"\n{fbasename}")
    
    for method in methods:
        res = Parallel(n_jobs = 25, verbose = 3)(delayed(run_sim)(seed=seed, spec=spec, method=method) for seed in range(Nsim))
        res_thetas[method] = np.vstack([x['thetas'] for x in res])
        res_shap[method] = np.vstack([x['shap'] for x in res])
    
    if print_MAE:
        for i, k in enumerate(true_thetas.keys()):
            if k != '000' and k != '111':
                res_str = "$\\langle " + ", ".join(k) + " \\rangle$ "
                for method in methods:
                    res_str += "& ${:.3f} \\pm {:.3f}$ ".format(np.mean(np.abs(res_thetas[method][:, i] - res_thetas['truth'][i])),
                                                        sem(np.abs(res_thetas[method][:, i] - res_thetas['truth'][i])))   
                res_str += "\\\\"
                print(res_str) 
        print("\midrule")
        for i in range(res_shap['truth'].shape[0]):
            res_str = "$\\mathrm{{SHAP}}_{}$ ".format(i)
            for method in methods:
                res_str += "& ${:.3f} \\pm {:.3f}$ ".format(np.mean(np.abs(res_shap[method][:, i] - res_shap['truth'][i])),
                                                    sem(np.abs(res_shap[method][:, i] - res_shap['truth'][i])))   
            res_str += "\\\\"
            print(res_str) 

    if print_bias:
        for i, k in enumerate(true_thetas.keys()):
            if k != '000' and k != '111':
                res_str = "$\\langle " + ", ".join(k) + " \\rangle$ "
                for method in methods:
                    res_str += "& ${:.3f}$ & ${:.3f}$ ".format(np.mean(res_thetas[method][:, i] - res_thetas['truth'][i]),
                                                        np.std(res_thetas[method][:, i]))   
                res_str += "\\\\"
                print(res_str) 
        print("\midrule")
        for i in range(res_shap['truth'].shape[0]):
            res_str = "$\\mathrm{{SHAP}}_{}$ ".format(i)
            for method in methods:
                res_str += "& ${:.3f}$ & ${:.3f}$ ".format(np.mean(res_shap[method][:, i] - res_shap['truth'][i]),
                                                    np.std(res_shap[method][:, i]))   
            res_str += "\\\\"
            print(res_str)

    if save:
        with open(fbasename + "thetas" + ".joblib", 'wb') as f:
            dump(res_thetas, f)
        
        with open(fbasename + "shap" + ".joblib", 'wb') as f:
            dump(res_shap, f)

    return res_thetas, res_shap

def main():
    pathlib.Path('results/').mkdir(exist_ok=True) 

    for spec in specs:
        _ = run_MC(spec)

if __name__ == "__main__":
    main()
