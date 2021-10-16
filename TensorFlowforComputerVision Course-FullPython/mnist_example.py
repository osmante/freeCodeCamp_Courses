import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import (Conv2D, Input, Dense, MaxPool2D,
                                    BatchNormalization, Flatten,
                                    GlobalAvgPool2D)

from deeplearning_models import functional_model, MyCustomModel
from my_utils import display_some_examples

# model creation approach
MODEL_APPROACH = "class_based"

# create a sequential model (sequential approach)
seq_model = tf.keras.Sequential(
    [
        Input(shape = (28, 28, 1)), # the inputs are grayscaled images (28 x 28)
        Conv2D(32, (3, 3), activation = 'relu'),
        Conv2D(64, (3, 3), activation = 'relu'),
        MaxPool2D(),
        BatchNormalization(),

        Conv2D(128, (3, 3), activation = 'relu'),
        MaxPool2D(),
        BatchNormalization(),

        GlobalAvgPool2D(),
        Dense(64, activation = 'relu'),
        Dense(10, activation = 'softmax') # 10 categories from 0 to 9
    ]
)

if __name__ == '__main__':

    # load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # print MNIST dataset shape for the train and test sets
    print("\nDataset shapes:")
    print("X train dataset shape: ", x_train.shape)
    print("Y train dataset shape: ", y_train.shape)
    print("x test  dataset shape: ", x_test.shape)
    print("y test  dataset shape: ", y_test.shape)

    # display some dataset examples
    if True:
        display_some_examples(x_train, y_train)
    
    # normalize the data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # expand the dimension of the data to fit it to the shape of the input
    x_train = np.expand_dims(x_train, axis = -1) # axis can also be 3 (last dim)
    x_test = np.expand_dims(x_test, axis = -1)

    # print MNIST dataset new shape
    print("\nDataset new shapes:")
    print("X train dataset shape: ", x_train.shape)
    print("Y train dataset shape: ", y_train.shape)
    print("x test  dataset shape: ", x_test.shape)
    print("y test  dataset shape: ", y_test.shape)

    # one-hot encode the labels (10 classes)
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # compile the model
    if MODEL_APPROACH == "sequential":
        model = seq_model
    elif MODEL_APPROACH == "functional":
        model = functional_model()
    elif MODEL_APPROACH == 'class_based':
        model = MyCustomModel()

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                  metrics = 'accuracy')

    # train the model
    print("\nTraining:\n")
    model.fit(x_train, y_train, batch_size = 64, epochs = 3,
              validation_split = 0.2)
    
    # evaluate the model on the test dataset
    print("\nTesting:\n")
    model.evaluate(x_test, y_test, batch_size = 64)