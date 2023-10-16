# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:32:07 2023

@author: exy029
"""

from rdkit import Chem
import numpy as np
from mordred import Calculator, descriptors, Autocorrelation, MoeType


class get_descriptors():#(smi='C', ignore_3D=True, name=None):
    '''
    

    Parameters
    ----------
    smi : STR
        Canonical SMILES. The default is 'C'=methane.
    ignore_3D : STR
        If 3d descriptors are computed. The default is True.
    name : STR, optional
        Name of the molecule . The default is None.

    Returns
    -------
    result_dict : Dict
        dictionary of molecular descriptors.
    
    keys : List
        List of molecular descriptors computed.

    '''
    def __init__(self, smi='C', 
                 ignore_3D=True, 
                 name=None):
        super(get_descriptors, self).__init__()
        self.name = name
        self.ignore_3D = ignore_3D
        self.smi = smi
        
        if self.name is not None:
            print('Compute the molecular descriptors of {}'.format(name))
    
    def evaluate(self):
        # create descriptor calculator with all descriptors
        molecular_D = Calculator(descriptors, ignore_3D=self.ignore_3D) # if ignore_3D=True , no 3d descriptors are computed
    
        print('Number of Molecular descriptors', len(molecular_D.descriptors))
        mol = Chem.MolFromSmiles(self.smi)

        # calculate descriptor value
        results = molecular_D(mol)
    
        # List of all molecular descripotrs computed
        keys = list(results.asdict().keys())

        # Delete missing values 
        results = results.drop_missing()

        # Create a dictionary
        result_dict = results.asdict()

        # Delete zero values and boolean descriptors 
        result_dict = {x:y for x, y in result_dict.items() if ( np.abs(y) > 1e-6 and y is not (True or False or None))}
    
        print('Number of Molecular descriptors after removing non-numerical values and zeros',len(result_dict))
    
        # Delete Autocorrelation descriptors: they may not directly
        # correspond to structural or physical properties of the molecule
        # Hollas B. An analysis of the autocorrelation descriptor for molecules. J Math
        # Chem 2003;33(2):91–101. http://dx.doi.org/10.1023/a:1023247831238
        Auto = Calculator(Autocorrelation)
        Auto_dict = Auto(mol).asdict()
        auto_list = list(Auto_dict.keys())
    
        for k in auto_list:
            result_dict = {x:y for x, y in result_dict.items() if x!=k}
        
        print('Number of Molecular descriptors after removing autocorrelation descriptors', len(result_dict))
        
        return result_dict , keys
