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

    segmentation = {'multivariate':False, 'split':0.8, 'validation_mode':False, 'segment_size':10, 'shuffle_data':False, 'overlap':0.0}
    compare = {'basic':True, 'linear':True} # Set comparison method/s
    weather_data_limits = (0,len(weatherData)) # for entire dataset, define (0,len(weatherData))
    plot_results = True
    load_prepared_arrays = False
    save_data = False

    # Neural network training params:
    prediction_method = "rnn" # options: rnn, lstm, gru
    n_steps = 20
    l_rate = 0.05


    #Step 3: Check if there are missing values

    print("CHECKING MISSING VALUES")
    print(weatherData.isnull().sum())


    #Step 4: Prepare data / Load data previously prepared
    """ Ideally, data should be prepared once and then saved.
    """

    folder = 'data_segment_size_'+str(segmentation['segment_size'])+'/'

    if segmentation['multivariate']:
        files_path = {'X_train':folder+'X_train_m.npy', 'X_valid':folder+'X_valid_m.npy', 'X_test':folder+'X_test_m.npy',
                      'Y_train':folder+'Y_train_m.npy', 'Y_valid':folder+'Y_valid_m.npy', 'Y_test':folder+'Y_test_m.npy'}
    else:
        files_path = {'X_train':folder+'X_train.npy', 'X_valid':folder+'X_valid.npy', 'X_test':folder+'X_test.npy',
                      'Y_train':folder+'Y_train.npy', 'Y_valid':folder+'Y_valid.npy', 'Y_test':folder+'Y_test.npy'}

    # register temperature minimu and maximum
    temperature_limits = []
    temperature_limits.append(weatherData['temperature'].min())
    temperature_limits.append(weatherData['temperature'].max())

    if load_prepared_arrays:
        prepared_data = load_numpy_arrays (files_path=files_path, validation_mode=segmentation['validation_mode'])
    else:
        weatherData['time'] = pd.to_datetime(weatherData['time'].str[:19]) # Format time column
        weather_df_norm = normalize(weatherData) # Normalize
        prepared_data = prepare_data(weather_df=weather_df_norm, weather_data_limits=weather_data_limits, segmentation=segmentation, save_data=save_data, files_path=files_path) # Segment and split data


    #Step 5: Prepare/Train and execute prediction

    if segmentation['overlap'] > 0.0:
        hour_factor = segmentation['segment_size'] * segmentation['overlap']
    else:
        hour_factor = segmentation['segment_size']
    train_test_model (prepared_data, temperature_limits, prediction_method, compare, segmentation, hour_factor, n_steps, l_rate, plot_results)


def normalize(df):
    """ Normalizes data to [0,1].
    """
    # copy the dataframe
    df_norm = df.copy()
    # apply min-max scaling
    for column in df_norm.columns:
        df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())

    return df_norm


def prepare_data(weather_df, weather_data_limits, segmentation, save_data, files_path):
    """ Splits dataset and segments time series.
    """
    print("\n\nSPLITTING AND SEGMENTING DATA -----------------------------------------------------")
    multivariate, split, validation_mode, segment_size, shuffle_data, overlap = segmentation['multivariate'], segmentation['split'], segmentation['validation_mode'], segmentation['segment_size'], segmentation['shuffle_data'], segmentation['overlap']
    overlap if overlap >= 0.0 else 0.0
    segment_shift = segment_size - round(segment_size * overlap) + 1
    ts_size = len(weather_df)

    # Segment time series
    data = weather_df.loc[weather_data_limits[0]:weather_data_limits[1]][['temperature','pressure','humidity']]
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
        assert len(X_data) == len(Y_data) ,"Array lengths don't match."
        p = np.random.permutation(len(X_data))
        X_data, Y_data = X_data[p], Y_data[p]

    # Split data into training/validation/test sets
    X_train, X_valid, X_test = X_data[:train_size], X_data[train_size:train_size+valid_size], X_data[train_size+valid_size:]
    Y_train, Y_valid, Y_test = Y_data[:train_size], Y_data[train_size:train_size+valid_size], Y_data[train_size+valid_size:]

    print("# samples for training: ", len(X_train))
    print("# samples for validation: ", len(X_valid))
    print("# samples for test: ", len(X_test))

    if save_data:
        np.save(files_path['X_train'], X_train)
        np.save(files_path['X_valid'], X_valid)
        np.save(files_path['X_test'], X_test)
        np.save(files_path['Y_train'], Y_train)
        np.save(files_path['Y_valid'], Y_valid)
        np.save(files_path['Y_test'], Y_test)

    if validation_mode:
        return X_train, X_valid, Y_train, Y_valid
    else:
        return X_train, X_test, Y_train, Y_test


def load_numpy_arrays (files_path, validation_mode):
    """ Loads previously saved numpy arrays containing segmented data
    """
    print("\n\nLOADING DATA -----------------------------------------------------")
    if validation_mode:
        X_train = np.load(files_path['X_train'])
        X_test = np.load(files_path['X_valid'])
        Y_train = np.load(files_path['Y_train'])
        Y_test = np.load(files_path['Y_valid'])
    else:
        X_train = np.load(files_path['X_train'])
        X_test = np.load(files_path['X_test'])
        Y_train = np.load(files_path['Y_train'])
        Y_test = np.load(files_path['Y_test'])

    print("# samples for training: ", len(X_train))
    print("# samples for test: ", len(X_test))

    return X_train, X_test, Y_train, Y_test


def basic_assumption(data, segmentation):
    """ Prediction method based in the stationarity assumption. Temperature in time t will be predicted as the same of time t-1.
    """
    print("\n\nBASIC ASSUMPTION -----------------------------------------------------\n")
    _, X_test, _, Y_test = data
    X_test_last = np.zeros_like(Y_test)
    for idx,item in enumerate(X_test):
        if segmentation['multivariate']:
            X_test_last[idx] = item[-1][0]
        else:
            X_test_last[idx] = item[-1]
    # MSE
    mse = mean_squared_error(Y_test, X_test_last)
    print("Mean squared error: %.8f" % mse)

    return mse


def linear_regression(data, segmentation, print_coef=False):
    """ Least squares Linear Regression
    """
    X_train, X_test, Y_train, Y_test = data
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
    coefficients = lreg.coef_
    if print_coef: print("Coefficients: ", coefficients)
    # MSE
    mse = mean_squared_error(Y_test, Y_pred)
    print("Mean squared error: %.8f" % mse)

    return mse, Y_pred, coefficients


def rnn(data, pred_linear, temperature_limits, n_steps, segmentation, hour_factor, l_rate, arquitecture='rnn', print_train=False):
    """ Implementation of the following recurrent neural network (RNN) architectures:
            - Basic RNN
            - Gated recurrent unit (GRU) RNN
            - Long short-term memory (LSTM) RNN
    """
    X_train, X_test, Y_train, Y_test = data
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    Y_train = torch.from_numpy(Y_train).float()
    Y_test = torch.from_numpy(Y_test).float()

    train_loss = []
    test_loss = []

    temperature_min, temperature_max = temperature_limits

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
                    outputs += [output]
                outputs = torch.stack(outputs, 1).squeeze(2)
                return outputs
        model = RNNModel()

    elif arquitecture=='lstm':
        print("\n\nLSTM -----------------------------------------------------\n")
        class LSTMModel(nn.Module):
            def __init__(self, n_hidden=20, n_cells=1):
                super(LSTMModel, self).__init__()
                self.n_hidden = n_hidden
                self.n_cells = n_cells
                self.lstm1 = nn.LSTMCell(1,self.n_hidden)
                if self.n_cells > 1:
                    self.lstm2 = nn.LSTMCell(self.n_hidden,self.n_hidden)
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
                    outputs += [output]
                outputs = torch.stack(outputs, 1).squeeze(2)
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
            out = out[:,-1] # Only use last output
            out = torch.reshape(out, (out.size(0),1)) # Only use last output
            loss = criterion(out, Y_train)
            train_loss.append(loss.item())
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
            pred = pred[:,-1] # Only use last output
            pred = torch.reshape(pred, (pred.size(0),1))
            loss = criterion(pred, Y_test)
            test_loss.append(loss.item())
            print("\nTest loss: %.8f" % loss.item())
            denorm_pred = np.array(pred.squeeze(1)) * (temperature_max - temperature_min) + temperature_min
            denorm_pred_limear = pred_linear.transpose()[0] * (temperature_max - temperature_min) + temperature_min
            denorm_Y_test = np.array(Y_test.squeeze(1)) * (temperature_max - temperature_min) + temperature_min
            plot_prediction(denorm_pred, denorm_pred_limear, denorm_Y_test, temperature_max, hour_factor, arquitecture, i)

    return train_loss, test_loss


def train_test_model(data, temperature_limits, prediction_method, compare, segmentation, hour_factor, n_steps, l_rate, plot_results):
    """ Execute models training and plot results
    """
    prediction_results = []

    if compare['basic']: error_basic = basic_assumption(data, segmentation=segmentation)

    if compare['linear']: error_linear, pred_linear, _ = linear_regression(data, segmentation=segmentation)

    if prediction_method == 'rnn':
        train_loss, test_loss = rnn(data, pred_linear, temperature_limits, segmentation=segmentation, hour_factor=hour_factor, n_steps=n_steps, l_rate=l_rate, arquitecture='rnn')
    elif prediction_method == 'lstm':
        train_loss, test_loss = rnn(data, pred_linear, temperature_limits, segmentation=segmentation, hour_factor=hour_factor, n_steps=n_steps, l_rate=l_rate, arquitecture='lstm')
    elif prediction_method == 'gru':
        train_loss, test_loss = rnn(data, pred_linear, temperature_limits, segmentation=segmentation, hour_factor=hour_factor, n_steps=n_steps, l_rate=l_rate, arquitecture='gru')
    else:
        print("Set one of the following methods for prediction: rnn, lstm, gru")

    prediction_results = error_basic, error_linear, train_loss, test_loss
    if plot_results: plot_graph(prediction_results, prediction_method, n_steps)


def plot_prediction(pred, pred_linear, Y_test, y_limit, hour_factor, prediction_method, step):
    """ Plot RNN temperature prediction compared to linear regression and target data
    """
    plt.figure(figsize=(20,4))
    plt.title('Time Series Prediction - Step '+str(step), fontsize=14)
    plt.xlabel('hours', fontsize=8)
    plt.ylabel('temperature (??C)', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlim(0, pred.size*hour_factor)
    plt.ylim(0, y_limit)
    assert pred.size == pred_linear.size == Y_test.size ,"Array lengths for x axis don't match."
    x_ticks = np.arange(pred.size)*hour_factor
    plt.plot(x_ticks, pred, color='r', alpha=0.75, linewidth = 0.3, label=prediction_method.upper())
    plt.plot(x_ticks, pred_linear, color='g', alpha=0.75, linewidth = 0.3, label='Linear regression')
    plt.plot(x_ticks, Y_test, color='k', alpha=0.25, linewidth = 2.0, label='Target data')
    plt.legend(loc='upper left')
    plt.savefig('predict_step_%d.png'%step, dpi=300)
    plt.close()


def plot_graph(prediction_results, prediction_method, n_steps):
    """ Plot RNN loss history
    """
    error_basic, error_linear, train_loss, test_loss = prediction_results[0], prediction_results[1], prediction_results[2], prediction_results[3]
    test_loss_large = []
    x_axis = []
    dim = len(train_loss)/len(test_loss) # Original dimension of RNN hidden layer
    for i in range(len(train_loss)):
        if (i+1) % dim == 0:
            test_loss_large.append(test_loss[int(i/dim)])
            x_axis.append(int((i+1) / dim))
        else:
            test_loss_large.append(None)
            x_axis.append(0)
    test_loss = test_loss_large
    x_length = len(prediction_results[2])
    plt.plot((error_basic,)*x_length, label='Basic assumption')
    plt.plot((error_linear,)*x_length, label='Linear regression')
    plt.plot(train_loss, label=prediction_method.upper()+' train')
    plt.plot(test_loss, label=prediction_method.upper()+' test', marker='o', markersize=4)
    plt.xlabel('epochs')
    plt.ylabel('mean squared error')
    plt.ylim(0, 0.001)
    plt.rc('xtick',labelsize=8)
    plt.rc('ytick',labelsize=8)
    plt.xticks(np.arange(0,x_length+dim,dim), np.arange(n_steps+1))
    plt.title("Prediction Loss")
    plt.legend()
    #plt.show()
    plt.savefig('predict.png')


if __name__=='__main__':
    main()

