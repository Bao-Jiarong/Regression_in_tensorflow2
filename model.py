'''
  Author       : Bao Jiarong
  Creation Date: 2020-06-17
  email        : bao.salirong@gmail.com
  Task         : Regression using Tensorflow 2
  Dataset      : winequality_red.csv
'''

import tensorflow as tf

class Block(tf.keras.models.Sequential):
    def __init__(self,n,m=1,activation="relu"):
        super().__init__()
        for i in range(m):
            self.add(tf.keras.layers.Dense(units              = n,
                                            activation         = activation,
                                            kernel_initializer = "glorot_uniform",
                                            bias_initializer   = "zeros"))

# class Reg(tf.keras.models.Sequential):
#     def __init__(self, classes):
#         super().__init__()
#         self.add(Block(n = 64, m = 2 ))
#         self.add(Block(n = classes, activation = "linear"))

class Reg(tf.keras.models.Sequential):
    def __init__(self, classes,filters=32):
        super().__init__()
        self.add(Block(n = filters * 4 ))
        self.add(Block(n = filters * 2 ))
        self.add(Block(n = filters * 1 ))
        self.add(Block(n = classes, activation = "linear"))
