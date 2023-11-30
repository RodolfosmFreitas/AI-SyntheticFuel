# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:14:00 2023

@author: exy029
"""
import numpy as np

import pandas as pd
import os
import sys
from descriptors_utils import read_descriptors
import argparse


# Reproducibility
np.random.seed(0)

# default to use cuda
parser = argparse.ArgumentParser(description='')
parser.add_argument('--case-name', type=str, default='Descriptors calculation from MORDRED', help='experiment name')
parser.add_argument('--data', type=str, default="../../DATA", help='data directory')
args = parser.parse_args()

print('------------ Arguments -------------')
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

# read fuel data [Name, SMILES, CN, METHOD]
# Hydrocarbons : n-alkanes, iso-alkanes, cycloalkanes, alkenes and aromatics
data    = pd.read_excel(args.data + '/CN_Hydrocarbons.xlsx').to_numpy()

# define the chemical sctructure of the fuels
fuel_list = list(data[:,0]) 

# Isomeric SMILES
smile_fuels = list(data[:,1])

# Get descriptors dataframe
X = read_descriptors(smile_fuels,fuel_list)

#Remove the descriptors with 95% of matching values
nunique = X.nunique()
cols_to_drop = nunique[nunique < 0.95*X.shape[0]].index
X = X.drop(cols_to_drop, axis=1)

# Save
X.to_excel('Descriptors_Hydrocarbons.xlsx', index=fuel_list,columns=list(X.keys()))

