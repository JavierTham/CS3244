import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.optimizers import SGD
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, AveragePooling2D, Dropout

import numpy as np
import matplotlib.pyplot as plt
import os

# Plotting visualisation
def model_perf_vis(history):
    
    history_dict = history.history
    train_loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    train_accuracy = history_dict['accuracy']
    val_accuracy = history_dict['val_accuracy']

    fig, axis = plt.subplots(ncols=1, nrows=2, figsize=(7,7))
    
    # Loss plot 
    
    epochs = range(1, len(val_loss_values) + 1)
    chart1 = sns.lineplot(ax=axis[0], x=epochs, y=train_loss_values, label='Training Loss')
    
    chart1 = sns.lineplot(ax=axis[0], x=epochs, y=val_loss_values, label='Validation Loss')
    chart1.set(xlabel='Epochs', ylabel='Loss')
    chart1.axes.set_title('Model Loss', fontsize=20)
    chart1.grid(which='major', axis='y')
    
    chart2 = sns.lineplot(ax=axis[1], x=epochs, y=train_accuracy, label='Training Accuracy')
    chart2 = sns.lineplot(ax=axis[1], x=epochs, y=val_accuracy, label='Validation Accuracy')
    chart2.set(xlabel='Epochs', ylabel='Accuracy')
    chart2.axes.set_title('Model Accuracy', fontsize=20)
    chart2.grid(which='major', axis='y')
    
    plt.tight_layout()
    plt.show()

num_epochs = 10

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

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

train_dataset = train_dataset.map(lambda x,y: (x/255, y))
val_dataset = val_dataset.map(lambda x,y: (x/255, y))
test_dataset = test_dataset.map(lambda x,y: (x/255, y))

classes = ['Chinee Apple', 'Lantana', 'Parkinsonia', 'Parthenium', 'Prickly Acacia', 'Rubber Vine', 'Siam Weed', 'Snake Weed', 'Negative']

# ---------------------------------------------------------------------------------------------------------------------------------------------------- #

###############
##  Model 1  ##
###############

model_1 = Sequential()
# 1st Convolution
model_1.add(Conv2D(filters=32, kernel_size = (3,3), activation='relu', input_shape=(256,256,3)))
model_1.add(MaxPooling2D((2,2)))

# 2nd Convolution
model_1.add(Conv2D(64, (3,3), activation='relu'))
model_1.add(MaxPooling2D((2,2)))

# 3rd Convolution
model_1.add(Conv2D(128, (3,3), activation='relu'))
model_1.add(MaxPooling2D((2,2)))

# 4th Convolution
model_1.add(Conv2D(128, (3,3), activation='relu'))
model_1.add(MaxPooling2D((2,2)))

# Flatten layer
model_1.add(Flatten())

# Fully connected layers
model_1.add(Dense(64, activation='relu'))
model_1.add(Dense(9, activation='softmax'))

model_1.compile('adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model_1.summary()

# ---------------------------------------------------------------------------------------------------------------------------------------------------- #

###############
##  Model 2  ##
###############

model_2 = Sequential()
# 1st Convolution
model_2.add(Conv2D(filters=32, kernel_size = (3,3), activation='relu',
                   padding='same',
                   kernel_initializer='he_normal', 
                   kernel_regularizer='l2',
                   input_shape=(256,256,3)))
model_2.add(MaxPooling2D((2,2)))

# 2nd Convolution
model_2.add(Conv2D(64, (3,3), activation='relu', padding='same',
                  kernel_initializer='he_normal', 
                  kernel_regularizer='l2'))
model_2.add(MaxPooling2D((2,2)))

# 3rd Convolution
model_2.add(Conv2D(128, (3,3), activation='relu', padding='same',
                  kernel_initializer='he_normal', 
                  kernel_regularizer='l2'))
model_2.add(MaxPooling2D((2,2)))

# 4th Convolution
model_2.add(Conv2D(128, (3,3), activation='relu', padding='same',
                  kernel_initializer='he_normal', 
                  kernel_regularizer='l2'))
model_2.add(MaxPooling2D((2,2)))

# Flatten layer
model_2.add(Flatten())

# Fully connected layers
model_2.add(Dense(64, activation='relu',
                  kernel_initializer='he_normal', 
                  kernel_regularizer='l2'))
model_2.add(Dense(9, activation='softmax'))

model_2.compile('adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model_2.summary()

# ---------------------------------------------------------------------------------------------------------------------------------------------------- #

###############
##  Model 3  ##
###############

model_3 = Sequential()

# 1st Convolution
model_3.add(Conv2D(16, (3, 3), activation='relu',input_shape=(256,256, 3)))
model_3.add(Conv2D(16, (3, 3), activation='relu'))
model_3.add(MaxPooling2D((2, 2)))

# 2nd Convolution
model_3.add(Conv2D(32, (3, 3), activation='relu'))
model_3.add(Conv2D(32, (3, 3), activation='relu'))
model_3.add(MaxPooling2D((2, 2)))

# 3rd Convolution
model_3.add(Conv2D(64, (3, 3), activation='relu'))
model_3.add(Conv2D(64, (3, 3), activation='relu'))
model_3.add(MaxPooling2D((2, 2)))

# 4th Convolution
model_3.add(Conv2D(128, (3, 3), activation='relu'))
model_3.add(Conv2D(128, (3, 3), activation='relu'))
model_3.add(MaxPooling2D((2, 2)))

# Flattened the layer
model_3.add(Flatten())

# Fully connected layers
model_3.add(Dense(64, activation='relu'))
model_3.add(Dense(9, activation='softmax'))

model_3.compile('adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model_3.summary()

# ---------------------------------------------------------------------------------------------------------------------------------------------------- #
