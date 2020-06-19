import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import cv2
import reg_layers
import loader1

np.random.seed(7)
tf.random.set_seed(7)
# np.set_printoptions(threshold=np.inf)

# Input/Ouptut Parameters
n_outputs  = 1
model_name = "models/real_estate/real_estate1"
data_path  = "data/real_estate.csv"

# Step 0: Global Parameters
epochs     = 100
lr_rate    = 0.005
batch_size = 64

# Step 1: Create Model
model = reg_layers.Reg(classes = n_outputs,filters=32)

# Step 2: Define Metrics
model.compile(optimizer= tf.keras.optimizers.RMSprop(learning_rate = lr_rate),
              loss     = "mse",
              metrics  = ['mae'])
# print(model.summary())
# sys.exit()

if sys.argv[1] == "train":
    # Step 3: Load data
    X_train, Y_train, X_test, Y_test,stats = loader1.load_data(data_path,True,0.9)
    stats.to_csv("data/house_stats.csv", sep=',', encoding='utf-8')

    # Step 4: Training
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    # Create a function that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = model_name,
                                                     save_weights_only=True,
                                                     verbose=0, save_freq="epoch")
    # model.load_weights(model_name)
    model.fit(X_train, Y_train,
              batch_size     = batch_size,
              epochs         = epochs,
              validation_data= (X_test,Y_test),
              callbacks      = [cp_callback])

    # Step 6: Evaluation
    mse,mae = model.evaluate(X_test, Y_test, verbose = 2)
    print("mse = ",mse,"mae = ",mae)

elif sys.argv[1] == "predict":
    # Step 3: Loads the weights
    model.load_weights(model_name)
    my_model = tf.keras.Sequential([model])

    # Step 4: Get Normalization values
    stats   = pd.read_csv("house.csv", sep = ',', header = 0)

    # Step 5: Prepare the input AND predict
    input = sys.argv[2].split(",")
    input = np.array([float(x) for x in input])
    input  = loader1.normalize(input, stats).values
    print(input)
    preds = my_model.predict(np.array([input]))
    print(preds)
