# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:49:43 2023

@author: exy029
"""

import matplotlib.pyplot as plt
from data_utils import remove_correlated_features, scaling, remove_low_variance
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
import argparse
import h5py
import time
from lazypredict.Supervised import LazyRegressor
import tables

# Reproducibility
np.random.seed(0)

# default to use cuda
parser = argparse.ArgumentParser(description='')
parser.add_argument('--case-name', type=str, default='Get features using Recursive Feature Elimination', help='experiment name')
parser.add_argument('--data', type=str, default="../../../DATA", help='data directory')
parser.add_argument('--scaling', type=str, default='Auto', help='How scaling the data: None, Auto, Range, Pareto, Log')
parser.add_argument('--train-size', type=float, default=0.75, help="% of data selected to train")
args = parser.parse_args()

print('------------ Arguments -------------')
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')


# read descritpors data 
# Hydrocarbons : n-alkanes, iso-alkanes, cycloalkanes, alkenes and aromatics
X                = pd.read_excel('Descriptors_Hydrocarbons.xlsx', index_col=0)

# read fuel data [Name, SMILES, CN, METHOD]
data    = pd.read_excel(args.data + '/CN_Hydrocarbons.xlsx').to_numpy()
fuel_list = data[:,0]

# Cetane Number 
y = np.array(data[:,2], dtype=float)

# plotting 
nalkanes        = y[:10]  
isoalkanes      = y[10:44]
alkenes         = y[44:66] 
aromatics       = y[66:97] 
cycloalkanes    = y[97:]

sizes= np.array([nalkanes.shape[0], isoalkanes.shape[0], alkenes.shape[0], aromatics.shape[0], cycloalkanes.shape[0]]) / y.shape[0]
labels = ['n-alkanes', 'iso-alkanes', 'alkenes', 'aromatics', 'cycloalkanes']

# Clustering the data based on the value of cetane number
n_clusters = 6
y_pred = KMeans(n_clusters=n_clusters).fit_predict(y[:,None])
u_labels = np.unique(y_pred)

plt.figure(figsize=(8,6),dpi=150)
plt.scatter(y, 
            y, 
            c=y_pred, 
            s=80,  
            label=r'Cluster 1')
plt.grid('True')
plt.box('True')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel(r'Cetane Number',fontsize=18)
plt.ylabel(r'Cetane Number',fontsize=18)
#plt.legend(loc='best', frameon=False, prop={'size': 16, 'weight': 'extra bold'})
plt.savefig('clustering_CN_kmeans.jpg', bbox_inches='tight', dpi=150)

idx1 = np.asarray(np.where(y_pred == 0)).squeeze()
idx2 = np.asarray(np.where(y_pred == 1)).squeeze()
idx3 = np.asarray(np.where(y_pred == 2)).squeeze()
idx4 = np.asarray(np.where(y_pred == 3)).squeeze()
idx5 = np.asarray(np.where(y_pred == 4)).squeeze()
idx6 = np.asarray(np.where(y_pred == 5)).squeeze()
idx  = [idx1, idx2, idx3 , idx4 , idx5, idx6]

# Remove correlated descriptors using Spearman Correlation
X = remove_correlated_features(X, y, threshold=0.9, method='spearman')
descriptors_list = list(X.keys())


# # Remove descriptors With variance lower than threshold
# X = remove_low_variance(X, threshold=0.01)

# Scaling the data 
X, y, X_ , X_std, y_, y_std = scaling(X, y, args)

X = X.to_numpy()

# Split train and test datasets
X_train, X_test, y_train, y_test = [], [], [], []
fuel_train, fuel_test = [], []
for i in range(n_clusters):
    X_t, X_tt, y_t, y_tt, idx_train, idx_test = train_test_split(X[idx[i]], y[idx[i]], np.arange(len(idx[i])),
                                                                  train_size=args.train_size, 
                                                                  shuffle=True, 
                                                                  random_state=0)
    X_train.append(X_t)
    X_test.append(X_tt)
    y_train.append(y_t)
    y_test.append(y_tt)
    fuel_train.append(fuel_list[idx_train])
    fuel_test.append(fuel_list[idx_test])
        
X_train = np.concatenate(X_train,axis=0)
X_test  = np.concatenate(X_test,axis=0)
y_train = np.concatenate(y_train,axis=0)
y_test  = np.concatenate(y_test,axis=0)

fuel_train = np.concatenate(fuel_train,axis=0)
fuel_test  = np.concatenate(fuel_test,axis=0)

X_train = pd.DataFrame(X_train, index=fuel_train, columns=descriptors_list)
X_test  = pd.DataFrame(X_test, index=fuel_test, columns=descriptors_list)

# Lazy Predict help build a lot of basic models without much code and 
# helps understand which models works better without any parameter tuning

reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)
models.to_hdf('models.hdf5', key='models', mode='w')


fig = plt.figure(figsize=(8,6),dpi=150)
plt.hist(y_train *y_std + y_, bins=6, histtype='barstacked', rwidth=0.5, color='black', alpha=0.5, label='Train')
plt.hist(y_test *y_std + y_, bins=6, histtype='barstacked', rwidth=0.5, color='blue', alpha=0.5, label='Test')
plt.legend(loc=5, frameon=False, prop={'size': 18, 'weight': 'extra bold'})
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel(r'Cetane Number',fontsize=18)
plt.ylabel(r'Counts',fontsize=18)
l, b, h, w = .6, .6, .25, .25
ax2 = fig.add_axes([l, b, w, h])
patches, texts, pcts = ax2.pie(sizes, labels=labels, colors=['black', 'gray', 'blue', 'green', 'magenta'], 
                               autopct='%1.1f%%')
plt.setp(pcts, color='white', fontsize=10, fontweight='bold')
plt.setp(texts, fontsize=12, fontweight=600)
plt.savefig('hist_CN.jpg', bbox_inches='tight', dpi=150)