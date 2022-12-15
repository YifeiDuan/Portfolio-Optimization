import numpy as np
import pandas as pd
import Models
import in_gp_modified_2 as in_gp
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import torch

def standardization(data):
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return mu, std, (data - mu)/std

def normalization(data):
    mean = np.mean(data, axis=0)
    _range_ = np.max(data, axis=0) - np.min(data, axis=0)
    return mean, _range_, (data - mean)/_range_

def LSTM_hyper_parameter_search(params_grids, output_dim, X_train, y_train, std_y, mu_y):
    """
    params_grids: dict {key:value = param(str):list[val1, val2, val3, ...]}
    """
    X_train_list = [X_train[:12000], X_train[12000:24000], X_train[24000:36000], X_train[36000:48000], X_train[48000:60000],
                   X_train[60000:72000], X_train[72000:84000]]
    y_train_list = [y_train[:12000], y_train[12000:24000], y_train[24000:36000], y_train[36000:48000], y_train[48000:60000],
                   y_train[60000:72000], y_train[72000:84000]]
    max_R2 = 0.0
    optimal_params = {'hidden_size':None, 'num_layers':None, 'learningRate':None, 'num_iterations':None}
    for hs in params_grids['hidden_size']:
        for nl in params_grids['num_layers']:
            for lr in params_grids['learningRate']:
                for ni in params_grids['num_iterations']:
                    R2_list = []
                    for k in range(len(X_train_list)):
                        X = X_train_list[k][:10000]
                        X_val = X_train_list[k][10000:]
                        y = y_train_list[k][:10000]
                        y_val = y_train_list[k][10000:]
                        LSTM = Models.LSTM(input_size=1, hidden_size=hs, num_layers=nl, output_size=output_dim, dropout=0)
                        LSTM.fit(X, y, learningRate=lr, num_iterations=ni, batch_size=128, weight_decay=0)
                    
                        y_val_pred = LSTM.predict(X_val).squeeze(-1)
                    
                        y_val_reform = y_val.cpu().detach().numpy()*std_y + mu_y
                        y_val_pred_reform = y_val_pred.cpu().detach().numpy()*std_y + mu_y
                    
                        R2_list.append(r2_score(y_val_reform, y_val_pred_reform))
                    R2 = np.mean(R2_list)
                    if R2 > max_R2:
                        max_R2 = R2_val
                        optimal_params['hidden_size'] = hs
                        optimal_params['num_layers'] = nl
                        optimal_params['learningRate'] = lr
                        optimal_params['num_iterations'] = ni
                        print("Update: optimal_params - {p}, R2 - {val}".format(p = optimal_params, val = max_R2))
    print("optimal_params: {p}".format(p = optimal_params))
    print("max R2_val: {val}".format(val = max_R2))
    return optimal_params