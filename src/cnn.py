import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.optimizers import SGD
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, AveragePooling2D, Dropout

import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

num_epochs = 10
pre = Precision()
re = Recall()
acc = CategoricalAccuracy()


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

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

from tensorflow.python.client import device_lib
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
print(get_available_devices()) 

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

train_dataset_128, val_dataset_128 = tf.keras.utils.image_dataset_from_directory(
    train_data_path,
    label_mode = "categorical",
    validation_split=0.2,
    subset = "both", # return both train and val datasets
    seed = 0,
    batch_size = 64,
    image_size=(128, 128)
)

test_dataset_128 = tf.keras.utils.image_dataset_from_directory(
    test_data_path,
    label_mode = "categorical",
    seed = 0,
    batch_size = 64,
    image_size=(128, 128)
)

train_dataset = train_dataset.map(lambda x,y: (x/255, y))
val_dataset = val_dataset.map(lambda x,y: (x/255, y))
test_dataset = test_dataset.map(lambda x,y: (x/255, y))

train_dataset_128 = train_dataset_128.map(lambda x,y: (x/255, y))
val_dataset_128 = val_dataset_128.map(lambda x,y: (x/255, y))
test_dataset_128 = test_dataset_128.map(lambda x,y: (x/255, y))

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

# Flatten layer
model_1.add(Flatten())

# Fully connected layers
model_1.add(Dense(64, activation='relu'))
model_1.add(Dense(9, activation='softmax'))

model_1.compile('adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model_1.summary()

hist_1 = model_1.fit(train_dataset, epochs = num_epochs, validation_data = val_dataset)

model_perf_vis(hist_1)

for batch in test_dataset.as_numpy_iterator():
    X, y = batch
    yhat = model_1.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f'Precision: {pre.result().numpy()}')
print(f'Recall: {re.result().numpy()}')
print(f'Accuracy: {acc.result().numpy()}')

model_1.save('M1.h5')

# ---------------------------------------------------------------------------------------------------------------------------------------------------- #

###############
##  Model 4  ##
###############

model_4 = Sequential()

# 1st Convolution
model_4.add(Conv2D(32, (3, 3), activation='relu',
                        input_shape=(256, 256, 3)))
model_4.add(MaxPooling2D((2, 2)))

# 2nd Convolution
model_4.add(Conv2D(64, (3, 3), activation='relu'))
model_4.add(MaxPooling2D((2, 2)))

# 3rd Convolution
model_4.add(Conv2D(128, (3, 3), activation='relu'))
model_4.add(MaxPooling2D((2, 2)))

# 4th Convolution
model_4.add(Conv2D(128, (3, 3), activation='relu'))
model_4.add(MaxPooling2D((2, 2)))

# Flattened the layer
model_4.add(Flatten())

# Fully connected layers
model_4.add(Dense(64, activation='relu'))
model_4.add(Dense(9, activation='softmax'))

model_4.compile('adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model_4.summary()

hist_4 = model_4.fit(train_dataset, epochs = num_epochs, validation_data = val_dataset)

model_perf_vis(hist_4)

for batch in test_dataset.as_numpy_iterator():
    X, y = batch
    yhat = model_4.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f'Precision: {pre.result().numpy()}')
print(f'Recall: {re.result().numpy()}')
print(f'Accuracy: {acc.result().numpy()}')

model_4.save('M4.h5')

# ---------------------------------------------------------------------------------------------------------------------------------------------------- #

model_4_128 = Sequential()

# 1st Convolution
model_4_128.add(Conv2D(32, (3, 3), activation='relu',
                        input_shape=(128, 128, 3)))
model_4_128.add(MaxPooling2D((2, 2)))

# 2nd Convolution
model_4_128.add(Conv2D(64, (3, 3), activation='relu'))
model_4_128.add(MaxPooling2D((2, 2)))

# 3rd Convolution
model_4_128.add(Conv2D(128, (3, 3), activation='relu'))
model_4_128.add(MaxPooling2D((2, 2)))

# 4th Convolution
model_4_128.add(Conv2D(128, (3, 3), activation='relu'))
model_4_128.add(MaxPooling2D((2, 2)))

# Flattened the layer
model_4_128.add(Flatten())

# Fully connected layers
model_4_128.add(Dense(64, activation='relu'))
model_4_128.add(Dense(9, activation='softmax'))

model_4_128.compile('adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model_4_128.summary()

hist_4_128 = model_4_128.fit(train_dataset_128, epochs = num_epochs, validation_data = val_dataset_128)

model_perf_vis(hist_4_128)

for batch in test_dataset_128.as_numpy_iterator():
    X, y = batch
    yhat = model_4_128.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f'Precision: {pre.result().numpy()}')
print(f'Recall: {re.result().numpy()}')
print(f'Accuracy: {acc.result().numpy()}')

model_4_128.save('M4_128.h5')
# ---------------------------------------------------------------------------------------------------------------------------------------------------- #
