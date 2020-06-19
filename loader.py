'''
  Author       : Bao Jiarong
  Creation Date: 2020-06-17
  email        : bao.salirong@gmail.com
  Task         : Regression using Tensorflow 2
  Dataset      : winequality_red.csv
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import tensorflow as tf
import random

np.random.seed(7)
tf.random.set_seed(7)
# np.set_printoptions(threshold=np.inf)


def normalize(data,stats):
    # return (data - stats['max']) / (stats['max'] - stats['min'])
    return (data - stats['mean']) / stats['std']

def load_data(filename, shuffle=True, split_ratio=0.8):
    df = pd.read_csv(filename,sep=';',header=0)
    print(df.isna().sum())
    df = df.dropna() # delete the rows that contain missing values

    # Normalize Data
    labels = df.pop('quality').values
    stats  = df.describe().transpose()
    df     = normalize(df,stats)
    points = df.values

    # Shuffle data
    if shuffle == True:
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        points = np.array(points)[indices]
        labels = np.array(labels)[indices]

    # Divide the data
    N = int(split_ratio * len(labels))

    return points[:N].astype(np.float32), \
           labels[:N].astype(np.float32), \
           points[N:].astype(np.float32), \
           labels[N:].astype(np.float32), \
           stats
