# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:14:12 2023

@author: exy029
"""

import pandas as pd
import os
import sys
import numpy as np
import logging
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import spearmanr, pearsonr


def remove_correlated_features(x, y, threshold=0.7, method='pearson'):
    '''
    Objective:
        Remove correlated features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model 
        to generalize and improves the interpretability of the model.

    Inputs: 
        x: features dataframe
        threshold: features with correlations greater than this value are removed, default=0.7
        method: Correlation method, default: pearson

    Output: 
        dataframe that contains only the non-highly-collinear features
    '''
    
    # Calculate the correlation between descriptors and the physical properties
    corr_ = np.zeros(x.shape[1])
    if method == 'pearson':
        for idx in range(x.shape[1]):
            corr_[idx],_ = pearsonr(x.to_numpy()[:,idx],y)
    elif method == 'spearman':
        for idx in range(x.shape[1]):
            corr_[idx],_ = spearmanr(x.to_numpy()[:,idx],y)
        
    # argsort descending order
    sort_values = abs(corr_).argsort()[::-1][:len(corr_)]

    x = x.reindex(columns=list(x.keys()[sort_values]))
    
    # Calculate the correlation matrix
    corr_matrix = x.corr('spearman')
    
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []
    
    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if np.abs(val) >= threshold:
                # Print the correlated features and the correlation value
                #print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns=drops)
    print('Removed Columns {}'.format(drops))
    
    return x

def remove_low_variance(x, threshold=0.1):
    selection = VarianceThreshold(threshold)
    selection.fit(x)
    return x[x.columns[selection.get_support(indices=True)]]

def scaling(X, Y, args):
    # Scaling the data
    if args.scaling == None:
        X_, X_std = 0, 1
        Y_, Y_std = 0, 1
        X = (X - X_)/ X_std
        Y = (Y - Y_)/ Y_std

    elif args.scaling == 'Auto':
        X_, X_std = X.mean(0), X.std(0)
        Y_, Y_std = Y.mean(0), Y.std(0)
        X = (X - X_)/ X_std
        Y = (Y - Y_)/ Y_std

    elif args.scaling == 'Range':
        X_, X_std = X.min(0), X.ptp(0)
        Y_, Y_std = Y.min(0), Y.ptp(0)
        X = (X - X_)/ X_std
        Y = (Y - Y_)/ Y_std

    elif args.scaling == 'Pareto':
        X_, X_std = X.mean(0), np.sqrt(X.std(0))
        Y_, Y_std = Y.mean(0), np.sqrt(Y.std(0))
        X = (X - X_)/ X_std
        Y = (Y - Y_)/ Y_std
        
    elif args.scaling == 'Log':
        X = np.log10(X)
        Y = np.log10(Y)
        X_, X_std = X.mean(0), X.std(0)
        Y_, Y_std = Y.mean(0), Y.std(0)
        X = (X - X_)/ X_std
        Y = (Y - Y_)/ Y_std

    return X, Y, X_ , X_std, Y_, Y_std

def load_data(X, Y, args):
    


    # Split train and test datasets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=args.train_size, shuffle=True)

    # Create a data loader to train the model
    data_train = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True)

    data_test = torch.utils.data.TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(Y_test))
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=True)


    
    print("total input data shape: {}".format(X_train.shape))
    print("total output data shape: {}".format(Y_train.shape))
    
    return X_train, Y_train, train_loader, X_test, Y_test, test_loader