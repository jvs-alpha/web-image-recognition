from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys

def loadimage(file):
    with open(file, "rb") as img:
        image = Image.open(img).convert("RGB")
        image = np.asarray(image)
        # This needs to be in this dimension for model to load it and process it coz we haave trained it with this dimension
        image = image.reshape((1, 300, 300, 3))
        return image


def imagecheck(model, image):
    image = loadimage(image)
    class_names = ['horse', 'human']
    print(class_names)
    normalization_layer = keras.layers.experimental.preprocessing.Rescaling(1./255)
    image_test = normalization_layer(image)
    model = keras.models.load_model(model)
    prediction = model.predict(image_test)
    return class_names[np.argmax(prediction)]

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Run python hvmtest.py <model> <type>")
        sys.exit(1)
    else:
        print(imagecheck(sys.argv[1], sys.argv[2]))
