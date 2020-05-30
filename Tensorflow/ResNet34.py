
from tensorflow import keras

class ResidualUnit(keras.layers.Layer):
  def __init__(self, filters, strides=1, activation='relu', **kwargs):
    super().__init__(**kwargs)
    self.activation = keras.activations.get(activation)
    self.main_layers= [
        keras.layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False),
        keras.layers.BatchNormalization(),
        self.activation,
        keras.layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False),
        keras.layers.BatchNormalization()]
    self.skip_layers = [keras.layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False),
                        keras.layers.BatchNormalization()] if strides > 1 else []
    
  def call(self, inputs):
    Z = inputs
    for layer in self.main_layers :
      Z = layer(Z)
    skipZ = Z;
    for layer in self.skip_layers :
      skipZ = layer(skipZ)
    return self.activation(Z + skipZ) 
                                  

def ResNet34()
  model = keras.models.Sequential()
  model.add(keras.layers.Conv2D(64, 7, strides=2, input_shape=[224, 224, 3],  padding="same", use_bias=False))
  model.add(keras.layers.BatchNormalization())
  model.add(keras.layers.Activation("relu"))
  model.add(keras.layers.MaxPool2D(pool_size=(3,3), strides=2, paddding='same'))       

  prev_filter=64
  for filters in [64]*3 + [128]*4 + [256]*6 + [512] *3:
    strides = 1 if prev_filter = filters else 2
    model.add(ResidualUnit(filters, strides)
    prev_filter = filters 

  model.add(keras.layers.GlobalAveragePool2D())
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(1000, activation="Softmax")


         
