# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 08:59:47 2023

@author: exy029
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_selection import SelectKBest, r_regression, mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score


def adjust_r2(r2, N, num_features):
    coef = (N - 1) / (N - num_features - 1) 
    return 1 - (1 - r2) * coef

# get a list of models to evaluate
def get_models(model, k):
    models = dict()
    for i in range(1,k+1):
        fs = RFE(estimator=model, n_features_to_select=i, step=1)
        models[str(i)] = Pipeline(steps=[('s',fs),('m',model)])
    return models


# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
    folds_scores = cross_val_score(model, 
                                   X, 
                                   y, 
                                   scoring='r2', 
                                   cv=cv, 
                                   n_jobs=2, 
                                   error_score='raise',
                                   verbose=0)
    fit = model.fit(X, y)
    return folds_scores, fit[:-1].get_feature_names_out()


def CrossValidationRFE(model, args, X, y, k):
    # get the models to evaluate
    models = get_models(model,k)
    
    # evaluate the models and store results
    results, names, features = list(), list(), list()
    for name, model in models.items():
        scores, selected_feat = evaluate_model(model, X, y)
        # Compute the adusted R2-score
        scores = adjust_r2(scores, X.shape[0], int(name))
        results.append(scores)
        names.append(name)
        features.append(selected_feat)
        print('> model %s Score mean %.3f  Score std (%.3f)' % (name, np.mean(scores), np.std(scores)))
        print('> selected features:', selected_feat)
    
    # plot model performance for comparison
    plt.figure(figsize=(8,6), dpi=150)
    plt.boxplot(results, labels=names, notch= True, 
                vert=True,
                boxprops=dict(color='black'),
                capprops=dict(color='black'),
                whiskerprops=dict(color='black'),
                flierprops=dict(color='black', markeredgecolor='black'),
                medianprops=dict(color='red'),)
    plt.ylabel(r'Adjusted $R^2$-score', fontsize=18)
    plt.xlabel(r'Number of Features', fontsize=18)
    plt.grid('True')
    plt.box('True')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(args.save_dir + '/feature_selection.jpg', bbox_inches='tight', dpi=150)
    
    return results, features

