from datetime import datetime
import os
import yaml
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from numpy import concatenate
import numpy as np
import pickle
from fire import Fire
from forecasting_class_plot_model import *


def evaluate_forecasts(train, actual, predicted, n, y_limits=None, plot_filename=None, model_name=None, save_plots=False):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train) - n, len(train)), train[-n:], label='Train', color='green')
    plt.plot(range(len(train), len(train) + len(actual)), actual, label='Actual', color='blue')
    plt.plot(range(len(train), len(train) + len(predicted)), predicted, label='Predicted', color='red')
    plt.xlabel('Time')
    plt.ylabel(model_name if model_name else 'Model Output')
    plt.legend()
    if y_limits:
        plt.ylim(y_limits)

    if save_plots and plot_filename:
        output_dir = os.path.dirname(plot_filename)
        os.makedirs(output_dir, exist_ok=True)

        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename_with_datetime = f"{plot_filename}_{model_name}_{current_datetime}_forecast.png"
        plt.savefig(filename_with_datetime)
        print(f"Forecast plot saved as {filename_with_datetime}")

    # Ensure the plot is displayed
    plt.show()

def transformer_block(input_layer, num_heads, key_dim, ff_dim, dropout_rate):
    # Multi-Head Attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(input_layer, input_layer)
    attention_output = Dropout(dropout_rate)(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(input_layer + attention_output)

    # Feed Forward with Alignment Layer
    aligned_out = Dense(ff_dim)(out1)  # Align the dimensions to ff_dim
    ff_output = Dense(ff_dim, activation='relu')(aligned_out)
    ff_output = Dropout(dropout_rate)(ff_output)
    out2 = LayerNormalization(epsilon=1e-6)(aligned_out + ff_output)

    return out2



def save_loss_plot(history, plot_filename, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{model_name} Loss over Epochs')

    if plot_filename:
        output_dir = os.path.dirname(plot_filename)
        os.makedirs(output_dir, exist_ok=True)
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename_with_datetime = f"{plot_filename}_{model_name}_{current_datetime}_loss.png"
        plt.savefig(filename_with_datetime)
        print(f"Loss plot saved as {filename_with_datetime}")

    plt.show()


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if isinstance(data, list) else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f'var{j+1}(t-{i})' for j in range(n_vars)]
    for i in range(n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [f'var{j+1}(t)' for j in range(n_vars)]
        else:
            names += [f'var{j+1}(t+{i})' for j in range(n_vars)]

    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def train_transformer(df, config):
    # Extract parameters from config
    prediction_period = config['model']['prediction_period']
    n_in = config['model']['n_in']
    n_out = config['model']['n_out']
    dropout_rate = config['model']['dropout_rate']
    learning_rate = config['training']['learning_rate']
    loss_function = config['training']['loss_function']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    num_heads = config['model'].get('num_heads', 4)
    key_dim = config['model'].get('key_dim', 64)
    ff_dim = config['model'].get('ff_dim', 256)

    # Data preprocessing
    values = df.values[:-prediction_period].astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled, n_in, n_out)

    values = reframed.values
    n_train_hours = values.shape[0] - prediction_period
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    # Reshape input for Transformer
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    # Model Definition
    input_shape = (train_X.shape[1], train_X.shape[2])
    inputs = Input(shape=input_shape)
    x = Dense(ff_dim)(inputs)  # Align input dimensions
    x = transformer_block(x, num_heads=num_heads, key_dim=key_dim, ff_dim=ff_dim, dropout_rate=dropout_rate)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)  # Linear output for regression
    model = Model(inputs, outputs)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer)

    # Training the Model
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size,
                        validation_data=(test_X, test_y), verbose=2, shuffle=False)

    # Save the loss plot
    save_loss_plot(history, config['evaluation']['loss_plot_filename'], model_name='Transformer')

    # Forecasting and Evaluation
    yhat = model.predict(test_X)  # Shape: (samples, timesteps, features)
    yhat_reshaped = yhat.reshape(yhat.shape[0], -1)  # Reshape to 2D for scaler
    inv_yhat = scaler.inverse_transform(yhat_reshaped)

    test_y = test_y.reshape((len(test_y), 1))  # Ensure test_y is 2D
    # Flatten test_X back to 2D for concatenation
    inv_y = concatenate((test_y, test_X[:, 1:].reshape(test_X.shape[0], -1)), axis=1)  # Align dimensions
    inv_y = scaler.inverse_transform(inv_y)[:, 0]  # Extract the first column after inverse scaling

    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print(f'Test RMSE: {rmse:.3f}')
    # Plot forecasts
    evaluate_forecasts(
        train=df.values[:-prediction_period],  # Training data
        actual=inv_y,  # Actual values
        predicted=inv_yhat,  # Predicted values
        n=config['model']['n'],  # Forecasting steps
        y_limits=config['evaluation']['y_limits'],
        plot_filename=config['evaluation']['plot_filename'],
        model_name=config['evaluation']['model_name'],
        save_plots=config['evaluation']['save_plots']
    )


def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    interval = config['data']['interval']
    path = config['data']['path']
    i = config['data']['index']

    loaded_dataframes, df_names = load_data(interval, path)

    df = loaded_dataframes[df_names[i]].copy()
    df.set_index(config['data']['index_column'], inplace=True)

    model_type = config['model'].get('model_type', 'lstm')
    if model_type == 'lstm':
        train_lstm(df, config)
    elif model_type == 'lstm_cnn':
        train_lstm_cnn(df, config)
    elif model_type == 'transformer':
        train_transformer(df, config)
    else:
        raise ValueError(f"Unsupported model_type '{model_type}' specified in config.")

if __name__ == '__main__':
    Fire(main)
