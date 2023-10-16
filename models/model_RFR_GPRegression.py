# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:36:44 2023

@author: exy029
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,  mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, explained_variance_score
import argparse
import pandas as pd
from data_utils import remove_correlated_features, scaling
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from feature_selection import CrossValidationRFE
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
import time
import os
import joblib
from sklearn.neighbors import KernelDensity
import probscale
import scipy.stats as st

# Reproducibility
np.random.seed(0)

# default to use cuda
parser = argparse.ArgumentParser(description='')
parser.add_argument('--case-name', type=str, default='Get features using Recursive Feature Elimination', help='experiment name')
parser.add_argument('--data', type=str, default="RandomForestRegressor", help='data directory')
parser.add_argument('--scaling', type=str, default='Auto', help='How scaling the data: None, Auto, Range, Pareto, Log')
parser.add_argument('--train-size', type=float, default=0.75, help="% of data selected to train")
parser.add_argument('--kernel', type=str, default='Matern', help='Output layer activation, RBF, Matern')
parser.add_argument('--nu', type=float, default=0.5, help="Matern parameter nu controlling the smoothness of the learned function ~ [0.5, 1.5,2.5, inf]")
parser.add_argument('--save-dir', type=str, default='Gaussian Process', help="Save directory")
args = parser.parse_args()

print('------------ Arguments -------------')
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir) 

# read data from recursive feature eleimination using random forest
X_train = pd.read_csv(args.data + '/X_train_RFE_RFR.csv', index_col=0)
X_test  = pd.read_csv(args.data + '/X_test_RFE_RFR.csv', index_col=0)
y_train = pd.read_csv(args.data + '/y_train.csv', index_col=0)
y_test  = pd.read_csv(args.data + '/y_test.csv', index_col=0)

descriptors_list = list(X_train.keys())
# re-scaling factors
X_      = pd.read_csv(args.data + '/X_mean_RFE_RFR.csv')
X_std   = pd.read_csv(args.data + '/X_std_RFE_RFR.csv')
y_      = np.loadtxt(args.data+'/y_mean.txt')
y_std   = np.loadtxt(args.data+'/y_std.txt')

fuel_train = list(y_train.index)
fuel_test  = list(y_test.index)

# Gaussian Process model
if args.kernel == 'RBF':
    kernel = RBF(length_scale=1.0)
elif args.kernel == 'Matern':
    kernel  = Matern(length_scale=1.0, nu=args.nu)
    
model   = GaussianProcessRegressor(kernel=kernel,
                                   random_state=0)
# fit the model
model.fit(X_train, y_train)

# feature importance
feat_imp_train = permutation_importance(model, 
                                  X_train, 
                                  model.predict(X_train),
                                  scoring='r2',
                                  n_repeats=10, 
                                  random_state=0, 
                                  n_jobs=2)

sorted_idx = feat_imp_train.importances_mean.argsort()

plt.figure(figsize=(8,6), dpi=150)
plt.boxplot(feat_imp_train.importances[sorted_idx].T,
            labels=np.array(descriptors_list)[sorted_idx],
            patch_artist=True,
            vert=False,
            boxprops=dict(facecolor='LightBlue'),
            )
plt.ylabel(r'Descriptors', fontsize=18)
plt.xlabel(r'Permutation importance', fontsize=18)
plt.grid('True')
plt.box('True')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(args.save_dir + '/feature_importance_train.jpg', bbox_inches='tight', dpi=150)

feat_imp_test = permutation_importance(model, 
                                  X_test, 
                                  model.predict(X_test),
                                  scoring='r2',
                                  n_repeats=10, 
                                  random_state=0, 
                                  n_jobs=2)

sorted_idx = feat_imp_test.importances_mean.argsort()

plt.figure(figsize=(8,6), dpi=150)
plt.boxplot(feat_imp_test.importances[sorted_idx].T,
            labels=np.array(descriptors_list)[sorted_idx],
            patch_artist=True,
            boxprops=dict(facecolor='LightGreen'),
            vert=False,
            )
plt.ylabel(r'Descriptors', fontsize=18)
plt.xlabel(r'Permutation importance', fontsize=18)
plt.grid('True')
plt.box('True')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(args.save_dir + '/feature_importance_test.jpg', bbox_inches='tight', dpi=150)

# save the model
joblib.dump(model, args.save_dir + '/GaussianProcessRegressor_model.pkl')

# load the model
#loaded_model = joblib.load(filename)

def scatter_plot(var1,var2, var3, var4, r2_train, r2_test):
    min_data  = np.minimum(np.minimum(np.amin(var1),np.amin(var2)), np.minimum(np.amin(var3), np.amin(var4)))
    max_data = np.maximum(np.maximum(np.amax(var1),np.amax(var2)), np.maximum(np.amax(var3),np.amax(var4)))
    
    plt.figure(figsize=(8,6),dpi=150)
    plt.scatter(np.reshape(var2,-1),np.reshape(var1,-1), 
                      s=50 ,marker='s', color = 'black', label=r'R2-score={:.3f}'.format(r2_train))
    plt.scatter(np.reshape(var3,-1),np.reshape(var4,-1), 
                      s=50 ,marker='v', color = 'red', label=r'R2-score={:.3f}'.format(r2_test))
    xlim = plt.xlim(min_data, max_data)
    ylim = plt.ylim(min_data, max_data)
    plt.plot(xlim,ylim,'r--',lw= 1)
    plt.grid('True')
    plt.box('True')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(r'$CN_{Predicted}$',fontsize=18)
    plt.ylabel(r'$CN_{True}$',fontsize=18)
    plt.legend(loc='best', frameon=False, prop={'size': 18, 'weight': 'extra bold'})
    plt.title('')
    plt.savefig(args.save_dir + '/statistical_error_scatter.jpg', bbox_inches='tight', dpi=150)


y_pred_test = model.predict(X_test) * y_std + y_
y_pred_train = model.predict(X_train) * y_std + y_
y_test = y_test * y_std + y_
y_train = y_train * y_std + y_

# Train
mape_train = mean_absolute_percentage_error(y_train, y_pred_train) 
mae_train = mean_absolute_error(y_train, y_pred_train)
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)
evar_train = explained_variance_score(y_train, y_pred_train)

# Test
mape_test = mean_absolute_percentage_error(y_test, y_pred_test) 
mae_test = mean_absolute_error(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
evar_test = explained_variance_score(y_test, y_pred_test)

# creata a dataframe
d = {'Train': [mape_train, mae_train, mse_train, r2_train, evar_train], #
     'Test': [mape_test, mae_test, mse_test, r2_test, evar_test]}

metrics = pd.DataFrame(data=d, index=['MAPE', 'MAE', 'MSE', 'R2', 'exp_Var'])

metrics.to_excel(args.save_dir + '/metrics.xlsx')

# Scatter plot
scatter_plot(y_train, y_pred_train, y_test, y_pred_test, r2_train, r2_test)

# Let's now compute the standardized errors
# Statistical diagnostics compare the predictive distribution to the distribution of the validation dataset
# Now, if our model is correct, the standarized errors must be distributed as a standard normal

'''
Diagnostics for Gaussian Process Emulators
Leonardo S. Bastos, Anthony O'Hagan
Technometrics, Vol. 51, No. 4, Special Issue on Computer Modeling (November 2009), pp. 425-438 (14 pages)
https://www.jstor.org/stable/40586652
Epsitemic Uncertainty
'''

e_train = (y_pred_train - y_train.to_numpy()[:,0]) / y_pred_train.std()
e_test = (y_pred_test - y_test.to_numpy()[:,0]) / y_pred_test.std()

zs = np.linspace(-3, 3, 100)

# kernel density estimation

kde_train = KernelDensity(kernel='gaussian', bandwidth="silverman").fit(e_train[:,None])
kde_test = KernelDensity(kernel='gaussian', bandwidth="silverman").fit(e_test[:,None])

log_dens_train = kde_train.score_samples(zs[:,None])
log_dens_test = kde_test.score_samples(zs[:,None])

plt.figure(figsize=(8,6), dpi=150)
plt.hist(e_train, 
          bins=20, 
          alpha=0.5, 
          density=True, 
          histtype='barstacked', 
          rwidth=0.8, 
          color='black')
plt.hist(e_test, 
          bins=20, 
          alpha=0.5, 
          density=True, 
          histtype='barstacked', 
          rwidth=0.8, 
          color='red')
plt.plot(zs, np.exp(log_dens_train), '-', color ='black', lw=2.0, label=r'Train')
plt.plot(zs, np.exp(log_dens_test), '-', color ='red', lw=2.0, label=r'Test')
plt.ylabel(r'PDF', fontsize=18)
plt.xlabel(r'Std. error', fontsize=18)
plt.xlim([-3,3])
plt.grid('True')
plt.box('True')
plt.legend(loc='best', frameon=False, prop={'size': 18, 'weight': 'extra bold'})
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(args.save_dir + '/histogram_error.jpg', bbox_inches='tight', dpi=150)

plt.figure(figsize=(8,6), dpi=150)
plt.plot(zs, np.exp(log_dens_train), '-', color ='black', lw=2.0, label=r'Train')
plt.plot(zs, np.exp(log_dens_test), '-', color ='red', lw=2.0, label=r'Test')
plt.ylabel(r'PDF', fontsize=18)
plt.xlabel(r'Std. error', fontsize=18)
plt.grid('True')
plt.box('True')
plt.legend(loc='best', frameon=False, prop={'size': 18, 'weight': 'extra bold'})
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(args.save_dir + '/PDF_error.jpg', bbox_inches='tight', dpi=150)

def equality_line(ax, label=None):
    limits = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.plot(limits, limits, 'k-', alpha=0.75, zorder=0, label=label)

scatter_options_train = dict(
    marker='s',
    markerfacecolor='black',
    markeredgecolor='black',
    markersize=8,
    markeredgewidth=1.25,
    linestyle='none',
    alpha=0.5,
    zorder=5,
    label='Train'
)


scatter_options_test = dict(
    marker='v',
    markerfacecolor='red',
    markeredgecolor='red',
    markersize=8,
    markeredgewidth=1.25,
    linestyle='none',
    alpha=0.5,
    zorder=5,
    label='Test'
)



fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
fig1 = probscale.probplot(e_train, 
                         ax=ax, 
                         dist=st.norm, 
                         plottype='qq',
                         probax='x',
                         #bestfit=True,
                         problabel='Theoretical Quantiles',
                         datalabel='Emperical Quantiles', 
                         labelsize=16,
                         scatter_kws=scatter_options_train, 
                         )

fig2 = probscale.probplot(e_test, 
                         ax=ax, 
                         dist=st.norm, 
                         plottype='qq',
                         probax='x',
                         #bestfit=True,
                         problabel='Theoretical Quantiles',
                         datalabel='Emperical Quantiles', 
                         scatter_kws=scatter_options_test, 
                         )
equality_line(ax, label='White Noise $\sim N(0,1)$')
plt.grid('True')
plt.box('True')
plt.legend(loc='lower right', frameon=False, prop={'size': 16, 'weight': 'extra bold'})
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel(r'Theoretical Quantiles',fontsize=18)
plt.ylabel(r'Empirical Quantiles',fontsize=18)
plt.savefig(args.save_dir + '/quantiles_error.jpg', bbox_inches='tight', dpi=150)
