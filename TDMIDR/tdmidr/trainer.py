#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
from sparsemax import Sparsemax
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def inverse_clr(X_clr):
    sparsemax_layer = Sparsemax(dim=1)
    return sparsemax_layer(X_clr)  


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def evaluate_model(predicted_values, X_RA, mask_tensor):


    reconstructed_ra =inverse_clr(predicted_values)
 
    mask_tensor = mask_tensor.to(torch.bool)  
    observed_values = X_RA[mask_tensor]  
    reconstructed_values = reconstructed_ra[mask_tensor]  
       
    observed_np = observed_values.detach().cpu().numpy()
    reconstructed_np = reconstructed_values.detach().cpu().numpy()

    
    
    mse_val = mean_squared_error(observed_np, reconstructed_np)
    mae_val = mean_absolute_error(observed_np, reconstructed_np)
    rmse_val = rmse(observed_np, reconstructed_np)

    
    return mse_val, mae_val, rmse_val

def train(model, X_train, X_ra, mask_train, mask_test, num_epochs=1000, lr=1e-2):
    optimizer_G = optim.Adam([model.G], lr=lr)
    optimizer_U = optim.Adam([model.U], lr=lr)
    optimizer_V = optim.Adam([model.V], lr=lr)
    optimizer_W = optim.Adam([model.W], lr=lr)
    optimizer_A = optim.Adam(model.A_list, lr=lr)

    total_losses = [] 
    parameter_losses = {name: [] for name in ['G', 'U', 'V', 'W', 'A']}  
       
    metrics_train = { 'mae': [], 'mse': [], 'rmse': [], 'r2': []}
    metrics_test = {'mae': [], 'mse': [],  'rmse': [], 'r2': []}
    val_losses = []  
    

    for epoch in range(num_epochs):
        model.train()
#         print(f"\n=== Epoch {epoch+1} ===")

        epoch_loss = 0.0

        for name, optimizer in zip(['G', 'U', 'V', 'W', 'A'],
                                   [optimizer_G, optimizer_U, optimizer_V, optimizer_W, optimizer_A]):
#         
            model.set_requires_grad(name)  

            optimizer.zero_grad()

            X_clr, mask = X_train.to(device), mask_train.to(device)
            loss = model.loss(X_clr, mask) 
            loss.backward()
            optimizer.step()
            
            
            loss_val = loss.item()
            parameter_losses[name].append(loss_val)  
            epoch_loss += loss_val
            
       
        avg_loss = epoch_loss / 5
        total_losses.append(avg_loss)


            
        _,mae_t, mse_t, rmse_t, _ = test(model, X_clr, X_ra, mask_train)
        
        metrics_train['mse'].append(mse_t)
        metrics_train['mae'].append(mae_t)
        metrics_train['rmse'].append(rmse_t)


           
        val_loss, val_mae, val_mse, val_rmse, _ = test(model, X_clr, X_ra, mask_test)
        metrics_test['mse'].append(val_mse)
        metrics_test['mae'].append(val_mae)
        metrics_test['rmse'].append(val_rmse)

        val_losses.append(val_loss)
        if (epoch + 1) % 200 == 0 or epoch == 0 or epoch == num_epochs - 1:

            print(f"Train  - MSE: {metrics_train['mse'][-1]:.4f}, MAE: {metrics_train['mae'][-1]:.4f}, "
                  f"RMSE: {metrics_train['rmse'][-1]:.4f}")
            print(f"Val - MSE: {metrics_test['mse'][-1]:.4f}, MAE: {metrics_test['mae'][-1]:.4f}, "
                  f"RMSE: {metrics_test['rmse'][-1]:.4f}")
            
    test_loss, test_mae, test_mse, test_rmse, test_results = test(model, X_clr, X_ra, mask_test)

    return test_results


def test(model, X_clr, X_ra, mask_val):
    model = model.to(device)
    model.eval()  # Switch to evaluation mode
    with torch.no_grad():
        # Forward pass
        X_hat, G_hat, U_hat, V_hat, W_hat, A_hat = model.forward()
        mse_val, mae_val, rmse_val = evaluate_model(X_hat, X_ra, mask_val)
        loss = model.loss(X_clr, mask_val)
        val_loss = loss.item()
        results = {
            "X_hat": X_hat,
            "G_hat": G_hat,
            "U_hat": U_hat,
            "V_hat": V_hat,
            "W_hat": W_hat,
            "A_hat": A_hat
        }

    return val_loss, mae_val, mse_val, rmse_val, results

