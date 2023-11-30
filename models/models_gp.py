#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 16:01:48 2023

@author: rodolfofreitas
"""
import gpytorch


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        
        self.mean_module = gpytorch.means.ConstantMean()
        
        if kernel == 'RBF':
            self.kernel = gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
        elif kernel == 'Linear':
            self.kernel = gpytorch.kernels.LinearKernel(ard_num_dims=train_x.shape[1])
        elif kernel == 'Matern1/2':
            self.kernel = gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=train_x.shape[1])
        elif kernel == 'Matern3/2':
            self.kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=train_x.shape[1])
        elif kernel == 'Matern5/2':
            self.kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[1])
        #
        self.covar_module = gpytorch.kernels.ScaleKernel(self.kernel, num_dims=train_x.shape[1])

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
