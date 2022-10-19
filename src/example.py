##########
# README #
##########

# Use this template to load the data
# DO NOT change the seed and validation_split
# You can edit ONLY the batch_size if you want

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2


import tensorflow as tf

train_data_path = os.path.join("..", "data", "train")
test_data_path = os.path.join("..", "data", "test")

train_dataset, val_dataset = tf.keras.utils.image_dataset_from_directory(
    train_data_path,
    label_mode = "categorical",
    validation_split=0.2,
    subset = "both", # return both train and val datasets
    seed = 0,
    batch_size = 64
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_data_path,
    label_mode = "categorical",
    seed = 0,
    batch_size = 64
)

if __name__ == "__main__":
    # all datasets are returned as tf.data.Dataset
    # index 0 is the image, index 1 is the one hot label
    for i in test_dataset:
        print(i[1].numpy())
        break