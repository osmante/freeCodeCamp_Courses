import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
import glob
import shutil
from sklearn.model_selection import train_test_split
import csv
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def display_some_examples(examples, labels):
    """
    Display the dataset examples (train/test)
    
    Parameters:
        examples: Train or test dataset (numpy array - uint8)
        labels: Train or test labels (numpy array - uint8)

    Returns:
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

def display_some_traffic_sign_examples(path):
    """
    Display the dataset examples (test)

    Parameters:
        path: test set path (str)

    Returns:
        None
    """

    images_paths = glob.glob(path + "\\*.png") # image paths
    plt.figure(figsize = (10, 10))

    for i in range(25):
        # select a random index in the dataset
        idx = np.random.randint(0, len(images_paths) - 1)
        img = imread(images_paths[idx])

        # subplot the dataset examples
        plt.subplot(5, 5, i + 1)
        plt.tight_layout(rect = [0, 0, 1, 0.95])
        plt.imshow(img)

    plt.suptitle("Dataset Examples")
    plt.show()         

def split_data(path_to_data, path_to_save_train,
               path_to_save_val, split_size = 0.1):
    """
    Split the data to get two different sets (train/valid, train/test)

    Parameters:
        path_to_data: to be split data path (str)
        path_to_save_train: to be saved train set path (str)
        path_to_save_val: to be saved validation set path (str)
        split_size: split ratio between the sets (float)

    Returns:
        None
    """
    
    folders = os.listdir(path_to_data)

    # get the data and split it
    for folder in folders:
        full_path = os.path.join(path_to_data, folder)
        images_paths = glob.glob(os.path.join(full_path, '*.png')) # image paths

        # split the data
        x_train, x_val = train_test_split(images_paths, test_size = split_size)

        for x in x_train:
            path_to_folder = os.path.join(path_to_save_train, folder)

            if not os.path.isdir(path_to_folder): # if the dir. not exist
                os.makedirs(path_to_folder) # create the directory
            
            print("Copying ", x, " to ", path_to_folder)
            shutil.copy(x, path_to_folder)

        for x in x_val:
            path_to_folder = os.path.join(path_to_save_val, folder)

            if not os.path.isdir(path_to_folder): # if the dir. not exist
                os.makedirs(path_to_folder) # create the directory
            
            print("Copying ", x, " to ", path_to_folder)
            shutil.copy(x, path_to_folder)

def order_test_set(path_to_images, path_to_csv, path_to_save_test):
    """
    Arrange the set set like the train set

    Parameters:
        path_to_images: test data path (str)
        path_to_csv: test data info path (str)
        path_to_save_test: to be saved test set path (str)

    Returns:
        None
    """

    try:
        with open(path_to_csv, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter = ',')

            for i, row in enumerate(reader):
                if i == 0: # continue the first line beacuse of the header line
                    continue

                img_name = row[-1].replace('Test/', '')
                label = row[-2]

                path_to_folder = os.path.join(path_to_save_test, label)

                if not os.path.isdir(path_to_folder): # if the dir. not exist
                    os.makedirs(path_to_folder)  # create the directory

                img_full_path = os.path.join(path_to_images, img_name)

                print("Copying ", img_full_path, " to ", path_to_folder)
                shutil.copy(img_full_path, path_to_folder)

    except:
        print("[INFO]: Error reading csv file.")

# preprocess and get the data
def create_generators(batch_size, train_data_path, 
                      val_data_path, test_data_path):
    
    preprocessor = ImageDataGenerator(
        rescale = 1 / 255.0
    )

    train_generator = preprocessor.flow_from_directory(
        train_data_path,
        class_mode = 'categorical',
        target_size = (60, 60),
        color_mode = 'rgb',
        shuffle = True,
        batch_size = batch_size
    )

    val_generator = preprocessor.flow_from_directory(
        val_data_path,
        class_mode = 'categorical',
        target_size = (60, 60),
        color_mode = 'rgb',
        shuffle = True,
        batch_size = batch_size
    )

    test_generator = preprocessor.flow_from_directory(
        test_data_path,
        class_mode = 'categorical',
        target_size = (60, 60),
        color_mode = 'rgb',
        shuffle = True,
        batch_size = batch_size
    )

    return train_generator, val_generator, test_generator