"""
Empirical application for the following paper:
Quintas-Martínez, V., Bahadori, M. T., Santiago, E., Mu, J. and Heckerman, D. 
"Multiply-Robust Causal Change Attribution" 
Proceedings of the 41st International Conference on Machine Learning, Vienna, Austria. PMLR 235, 2024.
"""

import pandas as pd
import numpy as np
import itertools
from statsmodels.stats.weightstats import DescrStatsW
from scipy.stats import norm
from math import comb
from mr_attribution import ThetaC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import  HistGradientBoostingClassifier, HistGradientBoostingRegressor
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pathlib

# Read and prepare data:
df = pd.read_csv('cps2015.csv')

# Integer encoding for education variables (works best for LightGBM):
educ_int = {'lhs' : 0, 'hsg' : 1, 'sc' : 2, 'cg' : 3, 'ad' : 4}
df['education'] = pd.from_dummies(df[['lhs', 'hsg', 'sc', 'cg', 'ad']]).replace(educ_int)

# Select variables:
df = df[['female', 'education', 'occ2', 'lnw', 'wage', 'weight']]
df.columns = ['female', 'education', 'occupation', 'lnwage', 'wage', 'weight']
df[['education', 'occupation']] = df[['education', 'occupation']].astype('category')

# Hyperparameters:
kwargs = {
    'regressor' : HistGradientBoostingRegressor, 
    'regressor_kwargs' : {'random_state' : 0},
    'classifier' : HistGradientBoostingClassifier,
    'classifier_kwargs' : {'random_state' : 0},
    'calibrator' : IsotonicRegression,
    'calibrator_kwargs' : {'out_of_bounds' : 'clip'},
}

# Split data into train and test set:
X = df[['education', 'occupation']].values
y = df['wage'].values
T = df['female'].values
w = df['weight'].values

kf = StratifiedKFold(n_splits = 2, shuffle = True, random_state = 0)
train_index, test_index = next(kf.split(X, T))

X_train, X_eval, y_train, y_eval, T_train, T_eval = X[train_index], X[test_index], y[train_index], y[test_index], T[train_index], T[test_index]
w_train, w_eval = w[train_index], w[test_index]

X_calib, X_train, _, y_train, T_calib, T_train, w_calib, w_train = train_test_split(X_train, y_train, T_train, w_train, 
                                                                                    train_size = 0.2, stratify = T_train, random_state = 0)

# Estimate scores
all_combos = [list(i) for i in itertools.product([0, 1], repeat=3)]
all_combos_minus1 = [list(i) for i in itertools.product([0, 1], repeat=2)]

scores = {}

for C in all_combos:
    scores[''.join(str(x) for x in C)] = ThetaC(C).est_scores(X_eval,
        y_eval,
        T_eval,
        X_train,
        y_train,
        T_train,
        w_eval=w_eval,
        w_train=w_train,
        X_calib=X_calib,
        T_calib=T_calib,
        w_calib=w_calib,
        **kwargs)

# Order sample weights in same way as scores:    
w_sort = np.concatenate((w_eval[T_eval==0], w_eval[T_eval==1])) 

# Function to compute Shapley values:
def compute_shapley(res_dict):
    shap = np.zeros(3)
    for k in range(3):
        sv = 0.0
        for C in all_combos_minus1:
            C1 = np.insert(C, k, True)
            C0 = np.insert(C, k, False)
            chg = (np.average(res_dict[''.join(map(lambda x : str(int(x)), C1))], weights=w_sort) - 
                    np.average(res_dict[''.join(map(lambda x : str(int(x)), C0))], weights=w_sort))
            sv += chg/(3*comb(2, np.sum(C)))
        shap[k] = sv
    return shap

# Function to compute standard errors by multiplier bootstrap:
def mult_boot(res_dict, Nsim=1000):
    thetas = np.zeros((8, Nsim))
    attr = np.zeros((3, Nsim))
    for s in range(Nsim):
        np.random.seed(s)
        new_scores = {}
        for k, x in res_dict.items():
            new_scores[k] = x + np.random.normal(0,1, X_eval.shape[0])*(x - np.average(x, weights=w_sort))
        thetas[:, s] = np.average(np.array([x for k, x in new_scores.items()]), axis=1, weights=w_sort)
        attr[:, s] = compute_shapley(new_scores)
    return np.std(attr, axis=1)

# Compute Shapley values and std. err.:    
shap, shap_se = compute_shapley(scores), mult_boot(scores)

# Plot Shapley values:
pathlib.Path('results/').mkdir(exist_ok=True) 
def star(x, y):
    if np.abs(x/y)<norm.ppf(.95):
        return ''
    elif np.abs(x/y)<norm.ppf(.975):
        return '*'
    elif np.abs(x/y)<norm.ppf(.995):
        return '**'
    else:
        return '***' 
nam = ["P(educ)", "P(occup | educ)", "P(wage | occup, educ)"]
crit = norm.ppf(.975)
stars = [star(x, y) for x, y in zip(shap, shap_se)]
fig, ax = plt.subplots()
ax.axvline(x = 0, color='lightgray', zorder=0)
fig.set_size_inches(7, 4)
color = 'C1' 
stats0 = DescrStatsW(y_eval[T_eval==0], weights=w_eval[T_eval==0], ddof=0)
stats1 = DescrStatsW(y_eval[T_eval==1], weights=w_eval[T_eval==1], ddof=0)
wagegap = (stats1.mean - stats0.mean)
wagegap_se = np.sqrt(stats1.std_mean**2 + stats0.std_mean**2)
ax.add_patch(Rectangle((0, 4.75), width = wagegap, height = 0.5, color=color, alpha=0.8))
ax.plot((wagegap-crit*wagegap_se, wagegap+crit*wagegap_se,), (5.0, 5.0), color='darkslategray', marker='|', solid_capstyle='butt')
ax.axhline(y = 5.0, color='lightgray', linestyle='dotted', zorder=0)
for i in range(3):
    pos = (shap[i], 3-i+0.25) if shap[i] < 0 else (0, 3-i+0.25)
    width = np.abs(shap[i])
    ax.add_patch(Rectangle(pos, width = width, height = 0.5, color=color, alpha=0.8))
    ax.axhline(y = 3+0.5-i, color='lightgray', linestyle='dotted', zorder=0)
    ax.plot((shap[i]-crit*shap_se[i], shap[i]+crit*shap_se[i]), (3-i+0.5, 3-i+0.5), color='darkslategray', marker='|', solid_capstyle='butt')
plt.yticks([5.0] + [3+0.5-i for i in range(3)], [f'Unconditional Wage Gap: {wagegap:.2f}*** ({wagegap_se:.2f})'] + 
           ["{}: {:.2f}{} ({:.2f})".format(nam[i], shap[i], stars[i], shap_se[i]) for i in range(3)])
plt.xlabel('Gender Wage Gap ($/hour)')
plt.savefig('results/shapley.pdf', bbox_inches='tight')

# Additional descriptive plots for Appendix:
w0, w1 = w_eval[T_eval==0], w_eval[T_eval==1]

data_male_eval = pd.DataFrame({'education' : X_eval[:,0][T_eval==0], 
                              'occupation' : X_eval[:,1][T_eval==0],
                              'wage' : y_eval[T_eval==0]})
data_female_eval = pd.DataFrame({'education' : X_eval[:,0][T_eval==1], 
                              'occupation' : X_eval[:,1][T_eval==1],
                              'wage' : y_eval[T_eval==1]})

educ_names = {0 : 'Less than HS', 1 : 'HS Graduate', 2 : 'Some College', 3 : 'College Graduate', 4 : 'Advanced Degree'}
data_male_eval['education'] = data_male_eval['education'].replace(educ_names)
data_female_eval['education'] = data_female_eval['education'].replace(educ_names)

cats_educ = [educ_names[i] for i in range(5)]

ind = np.arange(len(cats_educ))
share0, share1 = np.zeros(len(cats_educ)), np.zeros(len(cats_educ))
for i, c in enumerate(cats_educ):
    share0[i] = np.sum(w0*(data_male_eval['education'] == c))/np.sum(w0)*100
    share1[i] = np.sum(w1*(data_female_eval['education'] == c))/np.sum(w1)*100

fig = plt.figure()
fig.set_size_inches(6, 5)
plt.bar(ind, share0, 0.4, label='Male')
plt.bar(ind+0.4, share1, 0.4, label='Female')
plt.xticks(ind+0.2, cats_educ, rotation=20, ha='right')
plt.ylabel('Relative Frequency (%)')
plt.xlabel('Education')
plt.legend()
plt.savefig('results/education.pdf', bbox_inches='tight')

occup_names= {1 : 'Management', 2 : 'Business/Finance', 3 : 'Computer/Math', 4 : 'Architecture/Engineering', 5 : 'Life/Physical/Social Science', 
              6 : 'Community/Social Sevice', 7 : 'Legal', 8 : 'Education', 9 : 'Arts/Sports/Media', 10 : 'Healthcare Practitioner', 
              11 : 'Healthcare Support', 12 : 'Protective Services', 13 : 'Food Preparation/Serving', 14 : 'Building Cleaning/Maintenance', 
              15 : 'Personal Care', 16 : 'Sales', 17 : 'Administrative', 18: 'Farming/Fishing/Forestry', 19 : 'Construction/Mining', 
              20 : 'Installation/Repairs', 21 : 'Production', 22 : 'Transportation'}
data_male_eval['occupation'] = data_male_eval['occupation'].replace(occup_names)
data_female_eval['occupation'] = data_female_eval['occupation'].replace(occup_names)

cats_occu = ['Management', 'Sales', 'Administrative', 'Education', 'Healthcare Practitioner', 'Other']

ind = np.arange(len(cats_occu))
share0, share1 = np.zeros(len(cats_occu)), np.zeros(len(cats_occu))
for i, c in enumerate(cats_occu[:-1]):
    share0[i] = np.sum(w0*((data_male_eval['occupation'] == c) & (data_male_eval['education'] ==
                                               'College Graduate')))/np.sum(w0 * (data_male_eval['education'] ==
                                               'College Graduate'))*100
    share1[i] = np.sum(w1*((data_female_eval['occupation'] == c) & (data_female_eval['education'] ==
                                               'College Graduate')))/np.sum(w1 * (data_female_eval['education'] ==
                                               'College Graduate'))*100
share0[-1] = np.sum(w0*((~data_male_eval['occupation'].isin(cats_occu[:-1])) & (data_male_eval['education'] ==
                                               'College Graduate')))/np.sum(w0 * (data_male_eval['education'] ==
                                               'College Graduate'))*100
share1[-1] = np.sum(w1*((~data_female_eval['occupation'].isin(cats_occu[:-1])) & (data_female_eval['education'] ==
                                               'College Graduate')))/np.sum(w1 * (data_female_eval['education'] ==
                                               'College Graduate'))*100

fig = plt.figure()
fig.set_size_inches(6, 5)
plt.bar(ind, share0, 0.4, label='Male')
plt.bar(ind+0.4, share1, 0.4, label='Female')
plt.xticks(ind+0.2, cats_occu, rotation=20, ha='right')
plt.ylabel('Relative Frequency (%)')
plt.xlabel('Occupation | College Graduate')
plt.legend()
plt.savefig('results/occupation_condit.pdf', bbox_inches='tight')
