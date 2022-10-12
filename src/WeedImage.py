import tensorflow as tf
import os
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

save_dir = os.path.join("..", "data", "augmented_images")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    validation_split=0.2)
target_size = (224, 224)

# X - batch of images (batch_size, *target_size, channels)
# y - numpy array of labels (one-hot labels)
train_generator = train_datagen.flow_from_directory(
    "../data/",
    target_size=target_size,
    batch_size=1,
    seed=0,
    save_to_dir=save_dir,
    save_prefix="aug")

for data in train_generator:
    cv2.imshow("img", data[0].squeeze()) 
    cv2.waitKey()

    print("Label:", data[1])
    break