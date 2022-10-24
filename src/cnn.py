import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.optimizers import SGD
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, AveragePooling2D, Dropout

import numpy as np
import matplotlib.pyplot as plt
import os

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

model = Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(9, activation='sigmoid'))

model.compile('adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.summary()

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir)
hist = model.fit(train_dataset, epochs = num_epochs, validation_data = val_dataset, callbacks = [tensorboard_callback])

# Plotting performance
fig = plt.figure()
plt.plot(hist.history['loss'], color="teal", label = "loss")
plt.plot(hist.history['val_loss'], color="orange", label = "val_loss")
plt.suptitle("Loss", fontsize=20)
plt.legend(loc="upper left")
plt.show()

# Evaluate Performance
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
for batch in test_dataset.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')
'''
def loss(model, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)

    return loss_object(y_true=y, y_pred=y_)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

for epoch in range(NUM_EPOCHS):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # Training loop
    for iter in ds_train_batch:
        # Optimize the model
        x, y = iter[0], iter[1]
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Track progress
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss
        # Compare predicted label to actual label
        # training=True is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        epoch_accuracy.update_state(y, model(x, training=True))

    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 1 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

'''