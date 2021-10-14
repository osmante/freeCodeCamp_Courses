import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D,
                                    BatchNormalization, Flatten,
                                    GlobalAvgPool2D

# create a sequential 
model = tf.keras.Sequential(
    [
        Input(shape = (28, 28, 1)), # the inputs are grayscaled images (28 x 28)
        Conv2D(32, (3, 3), activation = 'relu'),
        Conv2D(64, (3, 3), activation = 'relu'),
        MaxPool2D(),
        BatchNormalization(),

        onv2D(128, (3, 3), activation = 'relu'),
        MaxPool2D(),
        BatchNormalization(),

        GlobalAvgPool2D(),
        Dense(64, activation = 'relu')
        Dense(10, activation = 'softmax') # 10 categories for from 0 to 9
    ]
)

def display_some_examples(examples, labels):
    """
    Display the dataset examples (train/test)
    
    Parameters:
        examples: Train or test dataset (numpy array - uint8)
        labels: Train or test labels (numpy array - uint8)

    Return:
        None
    """

    plt.figure(figsize = (10, 10))

    for i in range(25):
        # select a random index in the dataset
        idx = np.random.randint(0, examples.shape[0] - 1)
        img = examples[idx]
        label = labels[idx]

        # subplot the dataset examples
        plt.subplot(5, 5, i + 1)
        plt.title(f'Label: {label}')
        plt.tight_layout(rect = [0, 0, 1, 0.95])
        plt.imshow(img, cmap = 'gray')

    plt.suptitle("Dataset Examples")
    plt.show()

if __name__ == '__main__':

    # load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # print MNIST dataset shape for the train and test sets
    print("X train dataset shape: ", x_train.shape)
    print("Y train dataset shape: ", y_train.shape)
    print("x test  dataset shape: ", x_test.shape)
    print("y test  dataset shape: ", y_test.shape)

    # display some dataset examples
    display_some_examples(x_train, y_train)