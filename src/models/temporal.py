import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

class forecastRNN(nn.Module):
    def __init__(self, num_input, num_timesteps):
        super(forecastRNN, self).__init__()
        self.T = 1 if num_timesteps==0 else num_timesteps
        self.num_timesteps = num_timesteps
        
        # Autoencoder for feature prediction
        self.autoenc = nn.Sequential(
                nn.Linear(num_input, 1000),
                nn.BatchNorm1d(1000),
                nn.Dropout(p=0.5),

                nn.Linear(1000, 1000),
                nn.BatchNorm1d(1000),
                nn.Dropout(p=0.5),

                nn.Linear(1000, num_input),
                nn.BatchNorm1d(num_input)
                )

        self.rnn = nn.RNN(input_size = num_input,
                hidden_size = num_input, 
                num_layers = 2,
                batch_first=True,
                dropout=0.2, 
                bidirectional=False)
    
    def loss(self, ypred, y):
        #  print('Loss input = ', ypred.shape, y.shape)
        nelt = ypred.nelement()
        return nn.MSELoss()(ypred, y)*1./nelt

    def forward(self, x, t): 
        x_final = x[:, -1, :]
        x = x[:, :-1, :]
        (bsize, T, nfeat) = x.shape
        #  print('Input feat shape = ', x.shape)
        #  print('Output shape = ', x_final.shape)
        if self.T==1:
            return x[:, 0, :] 
        # Forward pass through the RNN
        h = self.rnn(x)[0]
        # Forward pass through the featue prediction model
        h = h.contiguous().view(bsize*T, nfeat)
        y = self.autoenc(h).view(bsize, T, nfeat)
        #  print('RNN output = {}, feat pred module output = {}'.format\
                #  (h.shape, y.shape))
        # Calculate the loss
        lossval = self.loss(y[:, :-1, :], x[:, 1:, :])
        #  print('Loss vals = ', lossval.shape, lossval.min(), lossval.max())
        gap = (t[:,-1] - t[:,-2]).view(-1, 1)
        #  print('Gaps = ', gap.shape, gap.min(), gap.max())
        
        x_hat = y[:, -1, :]
        x_all_gaps = torch.zeros([bsize, 6-T, nfeat])
        #  print('x_hat = ', x_hat.shape)
        for t_pred in range(6-T):
            h_hat = self.rnn(x_hat.unsqueeze(1))[0].squeeze()
            x_hat = self.autoenc(h_hat)
            x_all_gaps[:, t_pred, :] = x_hat
        gap = (gap - 1)[:,0].long()
        x_pred = x_all_gaps[range(bsize), gap, :]
        #  print('Final output = ', x_pred.shape)
        lossval += self.loss(x_pred, x_final)
        #  print('Loss vals = ', lossval.shape, lossval.min(), lossval.max())
        #  print(x_all_gaps.shape, x_pred.shape)
        return x_pred, lossval

class RNN(nn.Module):
    def __init__(self, num_input, num_timesteps):
        super(RNN, self).__init__()
        self.T = 1 if num_timesteps==0 else num_timesteps
        self.rnn = nn.RNN(input_size = num_input,
                hidden_size = num_input, 
                num_layers = 2,
                batch_first=True,
                dropout=0.2, 
                bidirectional=False)
        
    def forward(self, x): 
        if self.T==1:
            return x[:, 0, :] 
        x = self.rnn(x)[0]
        return x[:, -1, :]

class LSTM(nn.Module):
    def __init__(self, num_input, num_timesteps):
        super(LSTM, self).__init__()
        self.T = 1 if num_timesteps==0 else num_timesteps
        self.lstm = nn.LSTM(input_size = num_input,
                hidden_size = num_input, 
                num_layers = 3,
                batch_first=True,
                dropout=0.1, 
                bidirectional=False)
        
    def forward(self, x): 
        if self.T==1:
            return x[:, 0, :] 
        x = self.lstm(x)[0]
        return x[:, -1, :]


