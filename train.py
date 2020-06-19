'''
  Author       : Bao Jiarong
  Creation Date: 2020-06-17
  email        : bao.salirong@gmail.com
  Task         : Regression using Tensorflow 2
  Dataset      : winequality_red.csv
'''

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import loader
import model

np.random.seed(7)
tf.random.set_seed(7)
# np.set_printoptions(threshold=np.inf)

# Input/Ouptut Parameters
n_outputs  = 1
model_name = "models/wine/reg"
data_path  = "data/wine_quality.csv"

# Step 0: Global Parameters
epochs     = 100
lr_rate    = 0.002
batch_size = 32

model = model.Reg(classes = n_outputs,filters=32)

@tf.function
def my_mse(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    return mse

def my_mae(y_true, y_pred):
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    return mae

def my_msle(y_true, y_pred):
    msle = tf.reduce_mean(tf.square(tf.math.log(y_true + 1) - tf.math.log(y_pred + 1)))
    return msle

def my_mape(y_true, y_pred):
    mape = 100 * tf.reduce_mean(tf.abs(y_true - y_pred) / y_true)
    return mape

def my_logcosh(y_true, y_pred):
    logcosh = tf.math.log((tf.math.exp(y_pred - y_true) + tf.math.exp(y_true - y_pred))/2)
    return logcosh


# Step 2: Define Metrics
model.compile(optimizer= tf.keras.optimizers.RMSprop(learning_rate = lr_rate),
              loss     = my_mae,
              metrics  = ['mse'])
# print(model.summary())
# sys.exit()

if sys.argv[1] == "train":
    # Step 3: Load data
    X_train, Y_train, X_valid, Y_valid,stats = loader.load_data(data_path,True,0.8)
    stats.to_csv("data/wine_stats.csv", sep=',', encoding='utf-8')

    # Step 4: Training
    # Create a function that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = model_name,
                                                     save_weights_only=True,
                                                     verbose=0, save_freq="epoch")
    # model.load_weights(model_name)
    model.fit(X_train, Y_train,
              batch_size     = batch_size,
              epochs         = epochs,
              validation_data= (X_valid,Y_valid),
              callbacks      = [cp_callback])

    # Step 6: Evaluation
    mae, mse = model.evaluate(X_valid, Y_valid, verbose = 2)
    print("Evaluation, MAE: {:2.4f}, MSE: {:2.4f}".format(mae, mse))

elif sys.argv[1] == "predict":
    # Step 3: Loads the weights
    model.load_weights(model_name)
    my_model = tf.keras.Sequential([model])

    # Step 4: Get Normalization values
    stats   = pd.read_csv("data/wine_stats.csv", sep = ',', header = 0)

    # Step 5: Prepare the input AND predict
    input = sys.argv[2].split(",")
    input = np.array([float(x) for x in input])
    input  = loader.normalize(input, stats).values
    print(input)
    preds = my_model.predict(np.array([input]))
    print(preds[0])
