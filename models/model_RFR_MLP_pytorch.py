# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 09:09:43 2023

@author: exy029
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,  mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, explained_variance_score
import argparse
import pandas as pd
import time
import os
import scipy.stats as st
import probscale
from sklearn.neighbors import KernelDensity

import torch 
from nn import model, mse_loss
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
import shap 

# Reproducibility
np.random.seed(0)
torch.manual_seed(0)

# default to use cuda
parser = argparse.ArgumentParser(description='')
parser.add_argument('--case-name', type=str, default='Get features using Recursive Feature Elimination', help='experiment name')
parser.add_argument('--data', type=str, default="RandomForestRegressor", help='data directory')
parser.add_argument('--save-dir', type=str, default='Fully-Connected-NN', help="Save directory")

# NN
parser.add_argument('--neurons-fc', type=int, default=32, help='number of neurons in the fully-connected layer')
parser.add_argument('--num-layers', type=int, default=2, help='number of FC layers')
parser.add_argument('--n-epochs', type=int, default=1000, help='number of iterations to train (default: 1000)')
parser.add_argument('--lr', type=float, default=5e-3, help='initial learnign rate')
parser.add_argument('--weight-decay', type=float, default=5e-5, help="weight decay")
parser.add_argument('--activation', type=str, default='tanh', help='Hidden layer activation, [relu, elu, gelu, tanh, None=linear]')
parser.add_argument('--output-activation', type=str, default=None, help='Output layer activation, sigmoid ~ [0,1], None=linear, softplus ~ [0, inf[')
parser.add_argument('--log-interval', type=int, default=1, help='how many epochs to wait before logging training status')
parser.add_argument('--batch-size', type=int, default=80, help='input batch size for training')
args = parser.parse_args()

# Check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

#%% Fully-Connected model

# function to the test the model druing the training
def test():
    model.eval()
    mse = 0.
    denominator = 0.
    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_pred = model(x)
        
        mse += mse_loss(y, y_pred, reduction='sum').item()
        denominator += mse_loss(y, torch.mean(y, axis=0), reduction='sum').item()
        
    
    loss_t  = mse / len(test_loader.dataset)
    rmse_test = np.sqrt(loss_t)
    r2_score  = 1 - (mse / denominator) 
    return loss_t, rmse_test, r2_score

# Transform numpy array to float tensor
X_train = torch.FloatTensor(X_train.to_numpy())
X_test  = torch.FloatTensor(X_test.to_numpy())
y_train = torch.FloatTensor(y_train.to_numpy())
y_test  = torch.FloatTensor(y_test.to_numpy())

# Create a data loader to train the model
data_train = torch.utils.data.TensorDataset(X_train, torch.FloatTensor(y_train))
train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True)

data_test = torch.utils.data.TensorDataset(X_test, torch.FloatTensor(y_test))
test_loader = torch.utils.data.DataLoader(data_test, batch_size=X_test.shape[0], shuffle=True)


print("total input data shape: {}".format(X_train.shape))
print("total output data shape: {}".format(y_train.shape))

# Load the model
model = model(inp_dim=X_train.shape[1],
              out_dim=y_train.shape[1],
              n_layers=args.num_layers,
              neurons_fc=args.neurons_fc,
              hidden_activation=args.activation,
              out_layer_activation=args.output_activation)
print(model)
print("number of parameters {} of layers {}".format(*model.num_parameters()))

# Wrapper around our model to handle parallel training
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)

print("Start training network")
# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr,
                        weight_decay=args.weight_decay)

# Define the schedule
#scheduler = ExponentialLR(optimizer, gamma=0.99, verbose=True)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                    verbose=True, threshold=0.0001, threshold_mode='rel',
                    cooldown=0, min_lr=0, eps=1e-08)
tic = time.time()
start_time = tic
save_loss, save_loss_test = [], []
r2_train, r2_test, rmse_train, rmse_test = [], [], [], []

for epoch in range(1, args.n_epochs + 1):
    model.train()
    mse = 0.
    denominator = 0.
    ssim_ = 0.
    for batch_idx, (x, y) in enumerate(train_loader):
        
        x, y = x.to(device), y.to(device) 
        
        # forward propagation 
        model.zero_grad()
        y_pred = model(x)
            
        # Compute mse loss (HRR normalized) 
        loss = mse_loss(y, y_pred, reduction='sum')
            
        # Backward Step 
        loss.backward()
        optimizer.step()
        
        mse += loss.item()
        denominator += mse_loss(y, torch.mean(y, axis=0), reduction='sum').item()
        
    # Total loss
    loss_train = mse / len(train_loader.dataset)
    
    # Compute metrics 
    rmse  = np.sqrt(mse / len(train_loader.dataset))
    r2_   = 1 - (mse / denominator)
    
    # Check the model in the test data
    loss_test, rmse_t, r2_t = test()
    
    #  Save the losses
    save_loss.append(loss_train)
    save_loss_test.append(loss_test)
    
    # Scheduler the learning rate
    scheduler.step(rmse_t)
    #scheduler.step()
    
    # Disp the training process
    if epoch % args.log_interval == 0:
        # Save the metrics
        rmse_train.append(rmse)
        r2_train.append(r2_)
        rmse_test.append(rmse_t)
        r2_test.append(r2_t) 
        elapsed = time.time() - start_time
        print('It: %d, Training RMSE:%.4e, Training R2-score:%.4f, Time: %.3f' % (epoch, rmse, r2_, elapsed))
        print('Testing RMSE:%.4e, Testing R2-score:%.4f' % (rmse_t, r2_t))
        start_time = time.time()
            
    # save model
    if epoch == args.n_epochs:
        torch.save(model.state_dict(), args.save_dir +
                "/fcnn_model_iter{}.pth".format(epoch))

tic2 = time.time()
print("Done training {} epochs in {} seconds"
      .format(args.n_epochs, tic2 - tic))

# Save the training details
np.savetxt(args.save_dir + "/loss_train.txt", save_loss)
np.savetxt(args.save_dir + "/loss_test.txt", save_loss_test)
np.savetxt(args.save_dir + "/rmse_train.txt", rmse_train)
np.savetxt(args.save_dir + "/rmse_test.txt", rmse_test)
np.savetxt(args.save_dir + "/r2_train.txt", r2_train)
np.savetxt(args.save_dir + "/r2_test.txt", r2_test)



# Plot training and testing loss function and metrics
plt.figure(figsize=(8,6), dpi=150)
plt.plot(save_loss,'b-',lw=2, label='Train')
plt.plot(save_loss_test,'r--',lw=2, label='Test')
plt.box('True')
plt.grid('True')
plt.yscale("log")
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel(r'Loss',fontsize=18)
plt.xlabel(r'Number of epochs',fontsize=18)
plt.legend(loc='best', frameon=False, prop={'size': 18, 'weight': 'extra bold'})
plt.savefig(args.save_dir+'/losses_epoch{}.jpg'.format(args.n_epochs), bbox_inches='tight',dpi=150)


# feature importance using shapley values
explainer = shap.DeepExplainer(model, X_train)
shap_values = explainer.shap_values(X_train)
shap_mean  = np.abs(shap_values).mean(0)
sorted_idx = shap_mean.argsort()

plt.figure(figsize=(8,6), dpi=150)
plt.barh(np.array(descriptors_list)[sorted_idx],
         shap_mean[sorted_idx],
         color='LightBlue')
plt.ylabel(r'Descriptors', fontsize=18)
plt.xlabel('SHAP value  \n [Average impact on model output magnitude]', fontsize=18)
plt.grid('True')
plt.box('True')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(args.save_dir + '/shap_feature_importance_train.jpg', bbox_inches='tight', dpi=150)

explainer = shap.DeepExplainer(model, X_test)
shap_values = explainer.shap_values(X_test)
shap_mean  = np.abs(shap_values).mean(0)
sorted_idx = shap_mean.argsort()

plt.figure(figsize=(8,6), dpi=150)
plt.barh(np.array(descriptors_list)[sorted_idx],
         shap_mean[sorted_idx],
         color='LightGreen')
plt.ylabel(r'Descriptors', fontsize=18)
plt.xlabel('SHAP value  \n [Average impact on model output magnitude]', fontsize=18)
plt.grid('True')
plt.box('True')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(args.save_dir + '/shap_feature_importance_test.jpg', bbox_inches='tight', dpi=150)

######################## Make Predictions #####################################


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
    plt.xlabel(r'Predicted CN',fontsize=18)
    plt.ylabel(r'Exp. CN',fontsize=18)
    plt.legend(loc='best', frameon=False, prop={'size': 18, 'weight': 'extra bold'})
    plt.title('')
    plt.savefig(args.save_dir + '/statistical_error_scatter.jpg', bbox_inches='tight', dpi=150)


y_pred_test = model.predict(X_test) * y_std + y_
y_pred_train = model.predict(X_train) * y_std + y_
y_test = y_test * y_std + y_
y_train = y_train * y_std + y_

y_train = y_train.detach().numpy()
y_test  = y_test.detach().numpy()
y_pred_train = y_pred_train.detach().numpy()
y_pred_test = y_pred_test.detach().numpy()

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

e_train = (y_pred_train - y_train) / y_pred_train.std()
e_test = (y_pred_test - y_test) / y_pred_test.std()

zs = np.linspace(-3, 3, 100)

# kernel density estimation

kde_train = KernelDensity(kernel='gaussian', bandwidth="silverman").fit(e_train)
kde_test = KernelDensity(kernel='gaussian', bandwidth="silverman").fit(e_test)

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

