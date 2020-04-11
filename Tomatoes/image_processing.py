import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

data_dir = "dataset" # Image data directory name
categories = next(os.walk(data_dir))[1] # Gets a list of classes based on the folder names

valid_test_split = 0.1  # 10% training, 10% validation split 

img_size = 100

def pre_process_data():
    for category in categories: # Itterate over the folders

        path = os.path.join(data_dir,category) 
        counter = 0
        images = tqdm(os.listdir(path))
        no_img=len(images)  
        for img in images: # Itterate over the images of each folder
            if counter > (no_img * (1-(train_split*2))) and counter < (no_img * (1-(train_split*1))): 
                label = "valid"
            elif counter > (no_img * (1-(train_split*1))):
                label = "test"
            else:
                label = "train"
            try:
                img_array = cv2.imread(os.path.join(path,img))  
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                new_array = cv2.resize(img_array, (img_size, img_size)) # Resize the images 
                out_path = os.path.join("formated_data/"+label,category)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                cv2.imwrite(os.path.join(out_path,str(counter)+".jpg"),new_array) # save the images into the test, train, validation dir
            except Exception as e: 
                pass
            counter += 1

pre_process_data()
