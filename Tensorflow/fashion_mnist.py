
import tensorflow as tf
import os
from tensorflow import keras

root_logdir = os.path.join(os.curdir, "my_logs")

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

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(750, activation="relu"))
model.add(keras.layers.Dense(250, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

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
history = model.fit(X_train, y_train, epochs=20,
           validation_data=(X_valid, y_valid),
           callbacks=[tensorboard_cb])

import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
