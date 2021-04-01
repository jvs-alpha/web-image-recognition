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
        "train",
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
    print(len(image_test))
    if sys.argv[2] == "image":
        for i in range(20):
            index = i
            plt.grid(False)
            plt.imshow(image_test[i],cmap="gray")
            plt.xlabel("Actual: " + class_names[labels_test[index]])
            plt.title("Prediction " + class_names[np.argmax(prediction[index])])
            plt.show(block=False)
            plt.pause(3)
            plt.close()
    else:
        success = 0
        fail = 0
        for index in range(len(image_test)):
            if class_names[labels_test[index]] == class_names[np.argmax(prediction[index])]:
                success += 1
            else:
                fail += 1
        s_accu = (success / len(image_test))*100
        f_accu = (fail / len(image_test))*100
        print("Success Accuracy: {}".format(s_accu))
        print("Failure Accuracy: {}".format(f_accu))
