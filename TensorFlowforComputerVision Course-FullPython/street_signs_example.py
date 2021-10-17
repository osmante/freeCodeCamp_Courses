from my_utils import (split_data, order_test_set, 
                      display_some_traffic_sign_examples)
from deeplearning_models import streetsigns_model
from my_utils import create_generators
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

# for training set TRAIN to True, for testing set TRAIN to False
TRAIN = False

if __name__ == '__main__':

    # display some dataset examples
    if False:
        path = "D:\\Osman\\Datasets\\TrafficSignDataset\\Test"
        display_some_traffic_sign_examples(path)   

    # split the data
    if False:
        # directories
        path_to_data = "D:\\Osman\\Datasets\\TrafficSignDataset\\Train"
        path_to_save_train = "D:\\Osman\\Datasets\\TrafficSignDataset\\" \
                            "data\\train"
        path_to_save_val = "D:\\Osman\\Datasets\\TrafficSignDataset\\" \
                            "data\\val"

        # split the data
        split_data(path_to_data, path_to_save_train = path_to_save_train,
                path_to_save_val = path_to_save_val)
    
    # rearrange the test set to put it in the same order like the train set
    if False:
        path_to_images = "D:\\Osman\\Datasets\\TrafficSignDataset\\Test"
        path_to_csv = "D:\\Osman\\Datasets\\TrafficSignDataset\\Test.csv"
        path_to_save_test = "D:\\Osman\\Datasets\\TrafficSignDataset\\" \
                            "data\\test"
        order_test_set(path_to_images, path_to_csv, path_to_save_test)

    # create generators
    path_to_train = "D:\\Osman\\Datasets\\TrafficSignDataset\\" \
                    "data\\train"
    path_to_val = "D:\\Osman\\Datasets\\TrafficSignDataset\\" \
                  "data\\val"
    path_to_test = "D:\\Osman\\Datasets\\TrafficSignDataset\\" \
                   "data\\test"

    batch_size = 64
    epochs = 15
    lr = 0.001 # learning rate
    
    train_generator, val_generator, test_generator = \
        create_generators(batch_size, path_to_train, path_to_val, path_to_test)

    if TRAIN:
        # model save callback
        path_to_save_model = "./TrafficSignModels"
        ckpt_saver = ModelCheckpoint(
            path_to_save_model, # to be saved model file path
            monitor = 'val_accuracy', # monitor the validation accuracy
            mode = 'max', # check the monitored value is higher than prev. one
            save_best_only = True, # overwrite the previous save file
            save_freq = 'epoch', # check after each epoch ending
            verbose = 1 # show the debugging outputs
        )

        # early stop callback (after 10 epochs no increase, stop the training)
        early_stop = EarlyStopping(monitor = 'val_accuracy', patience  = 10)

        # compile the model
        nbr_classses = train_generator.num_classes

        model = streetsigns_model(nbr_classses)

        optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
        model.compile(optimizer = optimizer, loss = 'categorical_crossentropy',
                    metrics = ['accuracy'])
        model.fit(train_generator,
                epochs = epochs,
                batch_size = batch_size,
                validation_data = val_generator,
                callbacks = [ckpt_saver, early_stop]
                )
    # TEST
    else:
        model = tf.keras.models.load_model('./TrafficSignModels')
        model.summary()

        print("Evaluating validation set:\n")
        model.evaluate(val_generator)

        print("Evaluating test set:\n")
        model.evaluate(test_generator)