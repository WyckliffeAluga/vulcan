import numpy as np
import cv2
import tensorflow as tf
import os
import webbrowser

url = 'http://localhost:8080/index.html'
chrome_path = '/usr/bin/google-chrome %s'

data_dir = "dataset"
labels = next(os.walk(data_dir))[1]
labels.sort()

img_size = 256 

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read() # Read video stream
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert BGR color to RGB
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('p'): # On click of 'p' run prediction
        new_array = cv2.resize(img, (img_size, img_size))
        new_array = new_array.reshape(-1, img_size, img_size, 3)
        model = tf.keras.models.load_model("model.h5") # Load model
        cv2.imwrite("./app_data/test.jpg",new_array)
        prediction = model.predict(np.array(new_array)/255)
        pred_label = labels[int(np.where(prediction == np.amax(prediction))[0])]
        webbrowser.get(chrome_path).open(url+"?name="+pred_label) # Open up webpage for addtional details
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()