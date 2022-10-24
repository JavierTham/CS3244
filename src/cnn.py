import tensorflow as tf
import tensorflow_datasets as tfds

from keras.optimizers import SGD
from keras import datasets, layers, models
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
tf.random.set_seed(0)

# hyperparameters
NUM_EPOCHS = 1
BATCH_SIZE = 64
AUTOTUNE = tf.data.experimental.AUTOTUNE

(ds_train_batch, ds_val_batch, ds_test_batch), ds_info = tfds.load('deep_weeds',
                                                split=['train[:80]', 'train[80%:90%]', 'train[90%:]'],
                                                shuffle_files=True, as_supervised=True, with_info=True)

def normalise_img(img, label):
    return tf.cast(img, tf.float32)/255.0, label

ds_train_batch = ds_train_batch.map(normalise_img, num_parallel_calls=AUTOTUNE)
ds_train_batch = ds_train_batch.cache()
ds_train_batch = ds_train_batch.shuffle(ds_info.splits['train'].num_examples)
ds_train_batch = ds_train_batch.batch(BATCH_SIZE)
ds_train_batch = ds_train_batch.prefetch(AUTOTUNE)

ds_val_batch = ds_val_batch.map(normalise_img, num_parallel_calls=AUTOTUNE)
ds_val_batch = ds_val_batch.batch(128)
ds_val_batch = ds_val_batch.prefetch(AUTOTUNE)

model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(9, activation='sigmoid'))

optimizer = tf.keras.optimizers.SGD(lr=0.001)

model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.summary()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

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