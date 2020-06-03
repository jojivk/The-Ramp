
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

root_logdir = os.path.join(os.curdir, "my_logs")

class MyDense(keras.layers.Layer):
  def __init__(self, units, activation='relu', 
               kernel_initializer='glorot_normal', **kwargs) :
    super().__init__(**kwargs)
    self.units = units
    self.activation = keras.activations.get(activation)
    self.kinit = kernel_initializer

  def build(self, batch_input_shape):
    self.kernel = self.add_weight(name='kernel', 
                                  shape=[batch_input_shape[-1], self.units],
                                  initializer=self.kinit)
    self.bias   = self.add_weight(name='bias', 
                                  shape=[self.units], initializer='zeros')

    super().build(batch_input_shape)

  def call(self, X):
    return self.activation(X @ self.kernel + self.bias)

  def compute_output_shape(batch_input_shape):
    return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])

  def get_config():
    base_config = super.get_config()
    return {**base_config, "units": self.units,
            "activation": keras.activations.serialize(self.activation)}
    

class ResidualBlock(keras.layers.Layer):
  def __init__(self, in_dim, n_layers, **kwargs):
    super().__init__(**kwargs)
    self.layers = [MyDense(in_dim) for _ in range(n_layers)]

  def call(self, X):
    Z = X
    for layer in self.layers:
      Z = layer(Z)
    return tf.add(X, Z)

class ResidualModel(keras.Model):
  def __init__(self, input_shape, classes, **kwargs):
    super().__init__(**kwargs)
    self.flatten = keras.layers.Flatten(input_shape=input_shape)
    self.head    = MyDense(250)
    self.rblock  = ResidualBlock(250, 2)
    self.tail    = MyDense(classes, activation='softmax')

  def call(self, X):
   X = self.flatten(X)
   X = self.head(X)
   X = self.rblock(X)
   X = self.tail(X)
   return X



def get_run_logdir(root_logdir):
  import time
  run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
  return os.path.join(root_logdir, run_id)

logdir = get_run_logdir(root_logdir=root_logdir)

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test)= fashion_mnist.load_data()

X_valid, X_train = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

model = ResidualModel(input_shape=[28, 28], classes=10)

tensorboard_cb = keras.callbacks.TensorBoard(logdir)
#print(model.summary())
#print("Model Layers:", model.layers)
#
#hidden1 = model.layers[1]
#print("Layer 1 name:", hidden1.name)
#
#weights, biases = hidden1.get_weights()
#print(weights)
#print(biases)

opt = keras.optimizers.Adam(learning_rate=0.003)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10,
           validation_data=(X_valid, y_valid),
           callbacks=[tensorboard_cb])

import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
