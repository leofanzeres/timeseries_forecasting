import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
sys.path.append('utils/')
from prepareData import getWeatherData

def main():

    #Step 1: DownloadData

    weatherData = getWeatherData()
    print(weatherData)


    #Step 2: Define parameters

    segmentation = {'multivariate':False, 'split':0.8, 'validation_mode':False, 'segment_size':47, 'shuffle_data':True, 'overlap':0.0}
    prediction_method = "rnn" # options: basic, linear, rnn, lstm, gru

    # Neural network training params
    n_steps = 25
    l_rate = 0.05


    #Step 3: Check if there are missing values

    print("CHECKING MISSING VALUES")
    print(weatherData.isnull().sum())


    #Step 4: Create column for hour information

    weatherData['time'] = pd.to_datetime(weatherData['time'].str[:19])
    weatherData['hour'] = weatherData.index


    #Step 5: Normalize data

    weather_df_norm =  normalize(weatherData)


    #Step 6: Prepare/Train and execute prediction

    if prediction_method == 'basic':
        basic_assumption(weather_df_norm, segmentation=segmentation)
    elif prediction_method == 'linear':
        linear_regression(weather_df_norm, segmentation=segmentation)
    elif prediction_method == 'rnn':
        rnn(weather_df_norm, segmentation=segmentation, n_steps=n_steps, l_rate=l_rate, arquitecture='rnn')
    elif prediction_method == 'lstm':
        rnn(weather_df_norm, segmentation=segmentation, n_steps=n_steps, l_rate=l_rate, arquitecture='lstm')
    elif prediction_method == 'gru':
        rnn(weather_df_norm, segmentation=segmentation, n_steps=n_steps, l_rate=l_rate, arquitecture='gru')
    else:
        print("Set one of the following methods for prediction: basic, linear, rnn, lstm, gru")


def normalize(df):
    """ Normalizes data to [0,1].
    """
    # copy the dataframe
    df_norm = df.copy()
    # apply min-max scaling
    for column in df_norm.columns:
        df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())

    return df_norm


def segment_data(weather_df, segmentation):
    """ Splits dataset and segments time series.
    """
    print("\n\nSPLITTING AND SEGMENTING DATA -----------------------------------------------------")
    multivariate, split, validation_mode, segment_size, shuffle_data, overlap = segmentation['multivariate'], segmentation['split'], segmentation['validation_mode'], segmentation['segment_size'], segmentation['shuffle_data'], segmentation['overlap']
    overlap if overlap >= 0.0 else 0.0
    segment_shift = segment_size - round(segment_size * overlap) + 1
    ts_size = len(weather_df)


    # Segment time series
    data = weather_df.loc[:][['temperature','pressure','humidity','hour']]
    X_data, Y_data = [], []
    remain_size = ts_size
    while remain_size >= segment_size:
        start = ts_size - remain_size
        end = ts_size - remain_size + segment_size
        if multivariate:
            X_data.append(np.array(data[start:end][:]))
        else:
            X_data.append(np.array(data[start:end][:].temperature))
        Y_data.append(np.array(data[end:end+1][:].temperature))
        remain_size -= segment_shift
    X_data, Y_data = np.array(X_data), np.array(Y_data)
    size = len(X_data)
    train_size = int(size * split)
    valid_size = int((size - train_size)/2)
    test_size = size - train_size - valid_size

    # Shuffle arrays in unison
    if shuffle_data:
        if overlap > 0.0: print("\n\x1b[93mWarning: Since data is being shuffled, overlap should be 0.0. \nOtherwise overlapping segments may appear in different splits of the data.\033[0;0m\n")
        assert len(X_data) == len(Y_data) ,"Arrays' length does not match."
        p = np.random.permutation(len(X_data))
        X_data, Y_data = X_data[p], Y_data[p]

    # Split data into training/validation/test sets
    X_train, X_valid, X_test = X_data[:train_size], X_data[train_size:train_size+valid_size], X_data[train_size+valid_size:]
    Y_train, Y_valid, Y_test = Y_data[:train_size], Y_data[train_size:train_size+valid_size], Y_data[train_size+valid_size:]

    print("# samples for training: ", len(X_train))
    print("# samples for validation: ", len(X_valid))
    print("# samples for test: ", len(X_test))

    if validation_mode:
        X_test = X_valid
        Y_test = Y_valid

    return X_train, X_test, Y_train, Y_test


def basic_assumption(weather_df, segmentation):
    """ Prediction method based in the stationarity assumption. Temperature in time t will be predicted as the same of time t-1.
    """
    print("\n\nBASIC ASSUMPTION -----------------------------------------------------\n")
    segmentation['multivariate'] = False # Only temperature is necessary
    _, X_test, _, Y_test = segment_data(weather_df, segmentation)
    X_test_last = np.zeros_like(Y_test)
    for idx,item in enumerate(X_test):
        X_test_last[idx] = item[-1]
    # MSE
    print("Mean squared error: %.8f" % mean_squared_error(Y_test, X_test_last))


def linear_regression(weather_df, segmentation, print_coef=False):
    """ Least squares Linear Regression.
    """
    X_train, X_test, Y_train, Y_test = segment_data(weather_df, segmentation)
    if segmentation['multivariate']:
        X_train, X_test = X_train.reshape(X_train.shape[0], np.prod(X_train.shape[1:])), X_test.reshape(X_test.shape[0], np.prod(X_test.shape[1:]))
    print("\n\nLINEAR REGRESSION -----------------------------------------------------\n")
    # Instantiate linear regression object
    lreg = linear_model.LinearRegression()
    # Train the model
    lreg.fit(X_train, Y_train)
    # Make predictions using test set
    Y_pred = lreg.predict(X_test)

    # Coefficients
    if print_coef: print("Coefficients: ", lreg.coef_)
    # MSE
    print("Mean squared error: %.8f" % mean_squared_error(Y_test, Y_pred))


def rnn (weather_df, n_steps, segmentation, l_rate, arquitecture='rnn', print_train=False):
    """ Implementation of the following recurrent neural networks (RNN) architectures:
            - Basic RNN
            - Gated recurrent unit (GRU) RNN
            - Long short-term memory (LSTM) RNN
    """
    X_train, X_test, Y_train, Y_test = segment_data(weather_df, segmentation)
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    Y_train = torch.from_numpy(Y_train).float()
    Y_test = torch.from_numpy(Y_test).float()

    if arquitecture=='rnn' or arquitecture=='gru':
        print("\n\n"+arquitecture.upper()+" -----------------------------------------------------\n")
        class RNNModel(nn.Module):
            def __init__(self, hidden_dim=20, n_layers=1):
                super(RNNModel, self).__init__()
                self.hidden_dim = hidden_dim
                self.n_layers = n_layers
                if arquitecture=='rnn':
                    self.rnn = nn.RNN(1, self.hidden_dim, self.n_layers, bias=True, batch_first=True)
                elif arquitecture=='gru':
                    self.rnn = nn.GRU(1, self.hidden_dim, self.n_layers, bias=True, batch_first=True)
                self.linear = nn.Linear(self.hidden_dim, 1)

            def forward(self, x):
                outputs = []
                n_samples = x.size(0)
                h_t = torch.zeros((self.n_layers, n_samples, self.hidden_dim) , dtype=torch.float32)

                for input_t in x.split(1, dim=1):
                    input_t = input_t.unsqueeze(-1)
                    out_t, h_t = self.rnn(input_t, h_t)
                    output = self.linear(out_t).squeeze(1)
                    outputs.append(output)
                return outputs
        model = RNNModel()

    elif arquitecture=='lstm':
        print("\n\nLSTM -----------------------------------------------------\n")
        class LSTMModel(nn.Module):
            def __init__(self, n_hidden=20, n_layers=1, n_cells=1):
                super(LSTMModel, self).__init__()
                self.n_hidden = n_hidden
                self.n_layers = n_layers
                self.n_cells = n_cells
                self.lstm1 = nn.LSTMCell(1,self.n_hidden, self.n_layers)
                if self.n_cells > 1:
                    self.lstm2 = nn.LSTMCell(self.n_hidden,self.n_hidden, self.n_layers)
                self.linear = nn.Linear(self.n_hidden,1)

            def forward(self, x):
                outputs = []
                n_samples = x.size(0)
                h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
                c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
                if self.n_cells>1:
                    h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
                    c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
                for input_t in x.split(1, dim=1):
                    h_t, c_t = self.lstm1(input_t, (h_t, c_t))
                    if self.n_cells > 1:
                        h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
                        output = self.linear(h_t2)
                    else:
                        output = self.linear(h_t)
                    outputs.append(output)
                return outputs
        model = LSTMModel()

    else:
        print("Set one of the following arquitectures: rnn, lstm, gru")

    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(model.parameters(), lr=l_rate)
    for i in range(n_steps):
        if print_train:
            print("Step ", i)
        else:
            print("Step ", i, end='', flush=True)
        def closure():
            optimizer.zero_grad()
            if segmentation['multivariate']:
                input = X_train.view(X_train.size(0), -1)
                out = model(input)
            else:
                out = model(X_train)
            out = out[-1] # Only use last output
            loss = criterion(out, Y_train)
            if print_train:
                print("Train loss: %.8f" % loss.item())
            else:
                print('.', end='', flush=True)
            loss.backward()
            return loss
        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            if segmentation['multivariate']:
                input = X_test.view(X_test.size(0), -1)
                pred = model(input)
            else:
                pred = model(X_test)
            pred = pred[-1] # Only use last output
            loss = criterion(pred, Y_test)
            print("\nTest loss: %.8f" % loss.item())


if __name__=='__main__':
    main()
