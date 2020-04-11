import cv2
import tensorflow as tf
import os
import numpy as np

data_dir = "dataset"
labels = next(os.walk(data_dir))[1]
labels.sort()
print(labels)

def prepare(filepath):
    img_size = 256 
    img_array = cv2.imread(filepath) 
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    new_array = cv2.resize(img_array, (img_size, img_size))
    return new_array.reshape(-1, img_size, img_size, 3)


model = tf.keras.models.load_model("model.h5")
prediction = model.predict(np.array(prepare('tomoto_bac.jpeg'))/255)
print(" ------------ Prediction ---------------")
print(labels[int(np.where(prediction == np.amax(prediction))[0])])