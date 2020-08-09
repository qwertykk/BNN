import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import datasets
from keras.utils.vis_utils import plot_model
from keras import initializers
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects


def sign(x):
    result = tf.sign(x)
    return result


get_custom_objects().update({'custom_activation': Activation(sign)})

digit_data = datasets.mnist
(train_images, train_labels), (test_images, test_labels) = digit_data.load_data()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(784, activation=Activation(sign, name="SpecialActivation"),
                       bias_initializer=initializers.Constant(-1)),
    keras.layers.Dense(10, activation=tf.math.softmax)
])

for layer in model.layers:
    print(layer.name)
    # layer.set_weights([[np.sign(y) for y in x] for x in layer.get_weights()])
    print(layer.get_weights())

model.summary()

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_loss, test_acc)
