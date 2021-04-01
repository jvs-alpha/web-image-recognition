from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Run python hvmtest.py <model> <type>")
        sys.exit(1)
    test = keras.preprocessing.image_dataset_from_directory(
        "validation",
        seed=123,
        image_size=(300,300),
    )
    class_names = ['horses', 'humans']
    print(class_names)
    normalization_layer = keras.layers.experimental.preprocessing.Rescaling(1./255)
    normalized_ds = test.map(lambda x, y: (normalization_layer(x), y))
    image_test, labels_test = next(iter(normalized_ds))
    model = keras.models.load_model(sys.argv[1])
    test_loss, test_acc = model.evaluate(image_test,labels_test)
    prediction = model.predict(image_test)
