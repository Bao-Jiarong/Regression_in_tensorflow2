import tensorflow as tf
import pandas as pd
import numpy as np

def normalize(data,stats):
    # return (data - stats['max']) / (stats['max'] - stats['min'])
    return (data - stats['mean']) / stats['std']

def load_data(filename, shuffle=True, split_ratio=0.8):
    # Load Data
    df = pd.read_csv(filename,sep=',',header=0)
    print(df.isna().sum())
    df = df.dropna() # delete the rows that contain missing values

    # Normalize Data
    del df['No']
    # df['X1 transaction date'] = df['X1 transaction date'] - [int(x) for x in df['X1 transaction date']]
    df['X1 transaction date'] = df['X1 transaction date'] - df['X1 transaction date'].astype(np.int)
    df['X1'] = df['X1 transaction date']
    del df['X1 transaction date']
    df['X2'] = df['X2 house age']
    del df['X2 house age']

    df['X12'] = df['X1']**2
    df['X22'] = df['X2']**2
    # print(df)
    # df['X1/X2'] = (df['X1'])/(df['X2']+0.0001)
    # df['X1X2'] = (df['X1'])*(df['X2'])
    df['sinX1'] = tf.math.sin(df['X1']).numpy()
    df['sinX2'] = tf.math.sin(df['X2']).numpy()
    labels = df.pop('Y house price of unit area').values
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

    return points[:N], labels[:N], points[N:], labels[N:], stats
