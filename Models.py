import torch
from torch import nn

import gpytorch
import GPy
from gpytorch.constraints import Positive

import numpy as np
from torch.utils.data import DataLoader, TensorDataset

import warnings

"""
1. Neural Networks
"""
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=5, dropout=0):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1 # one-directional LSTM
        self.dropout = dropout
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout).to(self.device)
        self.linear = nn.Linear(self.hidden_size, self.output_size).to(self.device)

    def forward(self, input_seq):
        # input_seq (batch_size, seq_len=30, input_size=1)
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        pred = self.linear(output).to(self.device)  # (batch_size, seq_len, output_size=5)
        pred = pred[:, -1, :].to(self.device)  # (batch_size, output_size=5)
        return pred
    
    def fit(self, train_x, train_y, learningRate=0.002, num_iterations=1000, batch_size=128, weight_decay=0):        
        train_x = train_x.float().to(self.device) # (n_sample, seq_len=30, input_size=1)
        train_y = train_y.float().to(self.device) # (n_sample, output_size=5)
        dataset = TensorDataset(train_x, train_y)
        loader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learningRate, weight_decay=weight_decay)
        Loss = nn.MSELoss()
        
        for _ in range(num_iterations):
            for batch_x , batch_y in loader_train:
                predict_y = self.forward(batch_x)
                loss = Loss(predict_y, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
    def predict(self, x):
        self.eval()
        x = torch.tensor(x).float().to(self.device)
        pred = self.forward(x)
        return pred





"""
2. Gaussian Process
"""

"""
2.1 ExactGP with 30-day prices to predict 5-day returns. not time series analysis!

option 1: train_x are day 1~30 prices, train_y are 31~35 returns
"""

class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean, kernel):
        super(ExactGP, self).__init__(train_x, train_y, likelihood)
        
        if mean == "zero":
            self.mean_module = gpytorch.means.ZeroMean()
        elif mean == "linear":
            self.mean_module = gpytorch.means.LinearMean(input_size=train_x.shape[1])
        else:
            warnings.warn("You didn't specify a valid mean type")
        
        if kernel == "rbf":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]))
        elif kernel == "matern":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
        else:
            warnings.warn("You didn't specify a valid kernel type")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)   
        ## This is the prior for an abstract function f(x)

class GP():
    def __init__(self, train_x, train_y, mean="linear", kernel="rbf"):
        self.train_x = train_x.float()
        self.train_y = train_y.float()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGP(self.train_x, self.train_y, self.likelihood, mean, kernel)
    
    def fit(self, learningRate=0.01, num_iterations=1000, sigma=torch.tensor(0.1), seed=0):
        
        self.model.train()
        self.likelihood.train()
        
        if sigma.numel() == 1:
            self.model.likelihood.noise_covar.noise = sigma
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learningRate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        for _ in range(num_iterations):
            predict_y = self.model(self.train_x)
            #print("self.train_x.size: {size}".format(size = self.train_x.size()))
            #print("self.train_y.size: {size}".format(size = self.train_y.size()))
            #print("predict_y.mean.size: {size}".format(size = predict_y.mean.size()))
            loss = -1*mll(predict_y, self.train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def predict(self, x):
        self.model.eval()
        self.likelihood.eval()
        x = torch.tensor(x).float()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood( self.model(x) )
        
        return pred





"""
2.2 ExactINGP that parameterizes company_id & 30-day prices to predict 5-day returns

option 2: parameterize company id and day 1~30 prices, and then train_x are 31~35 time points, train_y are returns
"""
#import in_gp_modified as in_gp




"""
2.3 ExactGP with mere time series of returns

option 3: train_x are time points 1~35, train_y are respective returns, each company uses a separate model
"""
class IGP():
    def __init__(self, train_x, train_y, mean="zero", kernel="rbf"):
        self.train_x = train_x.float()  
        self.train_y = train_y.float()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.ids = torch.unique(train_x[:,0])  # the company ids

        self.likelihood_list = [gpytorch.likelihoods.GaussianLikelihood() for _ in range(len(self.ids))]
        self.model_list = [ExactGP(self.train_x[torch.where(self.train_x[:,0] == i)], train_y[torch.where(self.train_x[:,0] == i)],
                           self.likelihood_list[i], mean, kernel) for i in range(len(self.ids))]
    
    def fit(self, learningRate=0.01, num_iterations=1000, sigma=torch.tensor(0.1), seed=0):
        
        for model in self.model_list:
            model.train()
        for likelihood in self.likelihood_list:
            likelihood.train()
        
        if sigma.numel() == 1:
            for model in self.model_list:
                model.likelihood.noise_covar.noise = sigma
        
        optimizer_list = [torch.optim.Adam(self.model_list[i].parameters(), lr=learningRate) for i in range(len(self.ids))]
        mll_list = [gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood_list[i], self.model_list[i]) for i in range(len(self.ids))]
        
        for _ in range(num_iterations):
            for i in range(len(self.ids)):
                i_x = self.train_x[torch.where(self.train_x[:,0] == i)]
                i_y = self.train_y[torch.where(self.train_x[:,0] == i)]
                predict_y = self.model_list[i](i_x)
                loss = -1*mll_list[i](predict_y, i_y)
                optimizer_list[i].zero_grad()
                loss.backward()
                optimizer_list[i].step()
    
    def predict(self, x):
        for model in self.model_list:
            model.eval()
        for likelihood in self.likelihood_list:
            likelihood.eval()
        x = torch.tensor(x).float()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = []
            for i in range(x.shape[0]):
                idx = x[i,0]
                pred.append(self.likelihood_list[idx]( self.model_list[idx](x) )) 
        
        return torch.tensor(pred)