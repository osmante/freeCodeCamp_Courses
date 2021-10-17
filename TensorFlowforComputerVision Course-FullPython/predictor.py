import tensorflow as tf
import numpy as np

def predict_with_model(model, imgpath):
    """mnist_example.py"""

    image = tf.io.read_file(imgpath)
    image = tf.image.decode_png(image, channels = 3)
    image = tf.image.convert_image_dtype(image, tf.float32) # scales [0 - 1]
    image = tf.image.resize(image, [60, 60]) # (60, 60, 3)
    image = tf.expand_dims(image, axis = 0) # (1, 60, 60, 3)

    predictions = model.predict(image)
    prediction = np.argmax(predictions) # max probability index

    # because data generator read the data classes' name as string,
    # the predicted class name doesn't match with its order in the
    # predictions array. Example, prediction 2 deesn't mean the class 2, 
    # it means 10.
    # to resolve this problem a prediction dictionary can be created.
    classes_int = range(len(np.squeeze(predictions)))
    classes_str = [str(i) for i in classes_int]
    classes_str.sort()
    prediction_dict = {}

    for i, j in zip(classes_int, classes_str):
        prediction_dict[i] = int(j)

    prediction = prediction_dict[prediction]

    return prediction

if __name__ == '__main__':

    img_path = "D:\\Osman\\Datasets\\" \
               "TrafficSignDataset\\data\\test\\2\\00409.png"
    img_path = "D:\\Osman\\Datasets\\" \
               "TrafficSignDataset\\data\\test\\0\\00807.png"
    model = tf.keras.models.load_model('./TrafficSignModels')
    prediction = predict_with_model(model, img_path)

    print(f"prediction: {prediction}")