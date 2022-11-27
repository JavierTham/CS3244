# -*- coding: utf-8 -*-
"""kNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1p-_jcQ3Z16oGj_2GIPibtkApZHZBQB77
"""

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.optimizers import SGD
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, AveragePooling2D, Dropout

import numpy as np
import matplotlib.pyplot as plt
import os

train_data_path = os.path.join( "data", "train")
test_data_path = os.path.join("data", "test")

from google.colab import drive
drive.mount('/content/drive')

train_dataset1= tf.keras.utils.image_dataset_from_directory(
    "drive/MyDrive/CS3244/project/dataset/train",
    label_mode = "categorical",
    seed = 1,
    validation_split=0.2,
    subset = "training",
    batch_size = 64
)

valid_dataset1 = tf.keras.utils.image_dataset_from_directory(
    "drive/MyDrive/CS3244/project/dataset/train",
    label_mode = "categorical",
    seed = 1,
    validation_split=0.2,
    subset = "validation",
    batch_size = 64
)

train_dataset0= tf.keras.utils.image_dataset_from_directory(
    "drive/MyDrive/CS3244/project/dataset/train",
    label_mode = "categorical",
    seed = 0,
    validation_split=0.2,
    subset = "training",
    batch_size = 64
)

valid_dataset0 = tf.keras.utils.image_dataset_from_directory(
    "drive/MyDrive/CS3244/project/dataset/train",
    label_mode = "categorical",
    seed = 0,
    validation_split=0.2,
    subset = "validation",
    batch_size = 64
)


train_dataset2= tf.keras.utils.image_dataset_from_directory(
    "drive/MyDrive/CS3244/project/dataset/train",
    label_mode = "categorical",
    seed = 2,
    validation_split=0.2,
    subset = "training",
    batch_size = 64
)

valid_dataset2 = tf.keras.utils.image_dataset_from_directory(
    "drive/MyDrive/CS3244/project/dataset/train",
    label_mode = "categorical",
    seed = 2,
    validation_split=0.2,
    subset = "validation",
    batch_size = 64
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    "drive/MyDrive/CS3244/project/dataset/test",
    label_mode = "categorical",
    seed = 1,
    batch_size = 64
)

train_dataset0 = train_dataset0.map(lambda x ,y: ( x /255, y))
train_dataset1 = train_dataset1.map(lambda x ,y: ( x /255, y))
train_dataset2 = train_dataset2.map(lambda x ,y: ( x /255, y))
test_dataset = test_dataset.map(lambda x ,y: ( x /255, y))
valid_dataset0 = valid_dataset0.map(lambda x ,y: ( x /255, y))
valid_dataset1 = valid_dataset1.map(lambda x ,y: ( x /255, y))
valid_dataset2 = valid_dataset2.map(lambda x ,y: ( x /255, y))

training_data0 = list(train_dataset0.as_numpy_iterator())
training_data1 = list(train_dataset1.as_numpy_iterator())
training_data2 = list(train_dataset2.as_numpy_iterator())

testing_data = list(test_dataset.as_numpy_iterator())

validation_data0 = list(valid_dataset0.as_numpy_iterator())
validation_data1 = list(valid_dataset1.as_numpy_iterator())
validation_data2 = list(valid_dataset2.as_numpy_iterator())

def get_label_num (lst):
  index = 0
  for i in lst:
    if i == 1.0:
      return index
      break
    index += 1

def compress_row(lst):
  result_lst = []
  temp_lst = []
  new_num = 0
  if len(lst)%2 != 0:
    return lst
  else:
    for i in lst:
      temp_lst.append(i)
      if len(temp_lst) == 2:
        new_num = temp_lst[0] + temp_lst[1]
        result_lst.append(new_num)
        temp_lst = []
  return result_lst

def compress_column(matrix):
  result_lst = []
  temp_lst = []
  i = 0
  if len(matrix)%2 != 0:
    return matrix
  else:
    number_of_row = len(matrix)
    number_of_column = len(matrix[0])
    while i < number_of_row:
      for j in range(number_of_column):
        temp_lst.append(matrix[i][j])
        temp_lst.append(matrix[i+1][j])
      result_lst.append(compress_row(temp_lst))
      temp_lst = []
      i = i +2
    return result_lst

def compress_image(image):
  new_image = []
  new_row = []
  for row in image:
    new_row = compress_row(row)
    new_image.append(new_row)
  result_image = compress_column(new_image)
  return result_image

training_image_dict1 = {}
training_label_dict1 = {}
image = []
index = 0
for i in range(0, 75):
  for j in range(0, len(training_data1[i][0])):
    image = training_data1[i][0][j]
    new_image = compress_image(image)
    label = get_label_num(training_data1[i][1][j])
    training_image_dict1[index] = new_image
    training_label_dict1[index] = label
    index += 1

training_image_dict0 = {}
training_label_dict0 = {}
image = []
index = 0
for i in range(0, 75):
  for j in range(0, len(training_data0[i][0])):
    image = training_data0[i][0][j]
    new_image = compress_image(image)
    label = get_label_num(training_data0[i][1][j])
    training_image_dict0[index] = new_image
    training_label_dict0[index] = label
    index += 1
training_image_dict2 = {}
training_label_dict2 = {}
image = []
index = 0
for i in range(0, 75):
  for j in range(0, len(training_data2[i][0])):
    image = training_data2[i][0][j]
    new_image = compress_image(image)
    label = get_label_num(training_data2[i][1][j])
    training_image_dict2[index] = new_image
    training_label_dict2[index] = label
    index += 1

testing_image_dict = {}
testing_label_dict = {}
image = []
index = 0
for i in range(0, 15):
  for j in range(0, len(testing_data[i][0])):
    image = testing_data[i][0][j]
    new_image = compress_image(image)
    label = get_label_num(testing_data[i][1][j])
    testing_image_dict[index] = new_image
    testing_label_dict[index] = label
    index += 1

validation_image_dict0 = {}
validation_label_dict0 = {}
image = []
index = 0
for i in range(0, 19):
  for j in range(0, len(validation_data0[i][0])):
    image = validation_data0[i][0][j]
    new_image = compress_image(image)
    label = get_label_num(validation_data0[i][1][j])
    validation_image_dict0[index] = new_image
    validation_label_dict0[index] = label
    index += 1

validation_image_dict1 = {}
validation_label_dict1 = {}
image = []
index = 0
for i in range(0, 19):
  for j in range(0, len(validation_data1[i][0])):
    image = validation_data1[i][0][j]
    new_image = compress_image(image)
    label = get_label_num(validation_data1[i][1][j])
    validation_image_dict1[index] = new_image
    validation_label_dict1[index] = label
    index += 1

validation_image_dict2 = {}
validation_label_dict2 = {}
image = []
index = 0
for i in range(0, 19):
  for j in range(0, len(validation_data2[i][0])):
    image = validation_data2[i][0][j]
    new_image = compress_image(image)
    label = get_label_num(validation_data2[i][1][j])
    validation_image_dict2[index] = new_image
    validation_label_dict2[index] = label
    index += 1

import math
def distance_between_point(point1, point2):
  return (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2

def distance_between_line(line1, line2):
  sum_of_sq = 0
  for i in range(0, 128):
    sum_of_sq += distance_between_point(line1[i], line2[i])
  return sum_of_sq

def distance_between_images(image1, image2):
  sum_of_sq = 0
  for i in range(0, 128):
    sum_of_sq += distance_between_line(image1[i], image2[i])
  return math.sqrt(sum_of_sq)

def from_index_to_distance(i):
  return neighbour_distance_dict[i]

def find_knn_nearest_list1(k, i):
  neighbour_distance_dict = {}
  valid_image = validation_image_dict1[i]
  for n in training_image_dict1.keys():
    distance = distance_between_images(training_image_dict1[n], valid_image)
    neighbour_distance_dict[n] = distance
  index_lst = list(training_image_dict1.keys())
  index_lst.sort(key = lambda x: neighbour_distance_dict[x] )
  return index_lst[:k]

def find_knn_nearest_list2(k, i):
  neighbour_distance_dict = {}
  valid_image = validation_image_dict2[i]
  for n in training_image_dict2.keys():
    distance = distance_between_images(training_image_dict2[n], valid_image)
    neighbour_distance_dict[n] = distance
  index_lst = list(training_image_dict2.keys())
  index_lst.sort(key = lambda x: neighbour_distance_dict[x] )
  return index_lst[:k]

def find_knn_nearest_list0(k, i):
  neighbour_distance_dict = {}
  valid_image = validation_image_dict0[i]
  for n in training_image_dict1.keys():
    distance = distance_between_images(training_image_dict0[n], valid_image)
    neighbour_distance_dict[n] = distance
  index_lst = list(training_image_dict0.keys())
  index_lst.sort(key = lambda x: neighbour_distance_dict[x] )
  return index_lst[:k]

import random

def find_knn_nearest_answer1(lst):
  selected_dict = {}
  selected_label = 0
  max_num = 0
  result_list = []
  for m in range(9):
    selected_dict[m] = 0
  for index in lst:
    label = training_label_dict1[index]
    selected_dict[label] += 1
  for key, value in selected_dict.items():
    if value > max_num:
      max_num = value
      result_list = []
      result_list.append(key)
    elif value == max_num:
      result_list.append(key)
  if len(result_list) == 1:
    selected_label = result_list[0]
  else: 
    selected_label = random.choice(result_list)
  return selected_label

def find_knn_nearest_answer0(lst):
  selected_dict = {}
  selected_label = 0
  max_num = 0
  result_list = []
  for m in range(9):
    selected_dict[m] = 0
  for index in lst:
    label = training_label_dict0[index]
    selected_dict[label] += 1
  for key, value in selected_dict.items():
    if value > max_num:
      max_num = value
      result_list = []
      result_list.append(key)
    elif value == max_num:
      result_list.append(key)
  if len(result_list) == 1:
    selected_label = result_list[0]
  else: 
    selected_label = random.choice(result_list)
  return selected_label

def find_knn_nearest_answer2(lst):
  selected_dict = {}
  selected_label = 0
  max_num = 0
  result_list = []
  for m in range(9):
    selected_dict[m] = 0
  for index in lst:
    label = training_label_dict2[index]
    selected_dict[label] += 1
  for key, value in selected_dict.items():
    if value > max_num:
      max_num = value
      result_list = []
      result_list.append(key)
    elif value == max_num:
      result_list.append(key)
  if len(result_list) == 1:
    selected_label = result_list[0]
  else: 
    selected_label = random.choice(result_list)
  return selected_label

k = 3
predicted_validation_lists0k3 = []
for i in range(1187):
  knn_list = find_knn_nearest_list0(k,i)
  predicted_value = find_knn_nearest_answer0(knn_list)
  predicted_validation_lists0k3.append(predicted_value)
predicted_validation_lists1k3 = []
for i in range(1187):
  knn_list = find_knn_nearest_list1(k,i)
  predicted_value = find_knn_nearest_answer1(knn_list)
  predicted_validation_lists1k3.append(predicted_value)
predicted_validation_lists2k3 = []
for i in range(1187):
  knn_list = find_knn_nearest_list2(k,i)
  predicted_value = find_knn_nearest_answer2(knn_list)
  predicted_validation_lists2k3.append(predicted_value)

k = 5
predicted_validation_lists0k5 = []
for i in range(1187):
  knn_list = find_knn_nearest_list0(k,i)
  predicted_value = find_knn_nearest_answer0(knn_list)
  predicted_validation_lists0k5.append(predicted_value)
predicted_validation_lists1k5 = []
for i in range(1187):
  knn_list = find_knn_nearest_list1(k,i)
  predicted_value = find_knn_nearest_answer1(knn_list)
  predicted_validation_lists1k5.append(predicted_value)
predicted_validation_lists2k5 = []
for i in range(1187):
  knn_list = find_knn_nearest_list2(k,i)
  predicted_value = find_knn_nearest_answer2(knn_list)
  predicted_validation_lists2k5.append(predicted_value)

k = 7
predicted_validation_lists0k7 = []
for i in range(1187):
  knn_list = find_knn_nearest_list0(k,i)
  predicted_value = find_knn_nearest_answer0(knn_list)
  predicted_validation_lists0k7.append(predicted_value)
predicted_validation_lists1k7 = []
for i in range(1187):
  knn_list = find_knn_nearest_list1(k,i)
  predicted_value = find_knn_nearest_answer1(knn_list)
  predicted_validation_lists1k7.append(predicted_value)
predicted_validation_lists2k7 = []
for i in range(1187):
  knn_list = find_knn_nearest_list2(k,i)
  predicted_value = find_knn_nearest_answer2(knn_list)
  predicted_validation_lists2k7.append(predicted_value)

def calculate_accuracy(list1, list2):
  len1 = len(list1)
  len2 = len(list2)
  accurate_num = 0
  list_len = len1
  if len2 < len1:
    list_len = len2
  for i in range(list_len):
    if list1[i] == list2[i]:
      accurate_num += 1
  return accurate_num/list_len

accuracys0k3 = calculate_accuracy(predicted_validation_lists0k3, list(validation_label_dict0.values()))
accuracys1k3 = calculate_accuracy(predicted_validation_lists1k3, list(validation_label_dict1.values()))
accuracys2k3 = calculate_accuracy(predicted_validation_lists2k3, list(validation_label_dict2.values()))
accuracys0k5 = calculate_accuracy(predicted_validation_lists0k5, list(validation_label_dict0.values()))
accuracys1k5 = calculate_accuracy(predicted_validation_lists1k5, list(validation_label_dict1.values()))
accuracys2k5 = calculate_accuracy(predicted_validation_lists2k5, list(validation_label_dict2.values()))
accuracys0k7 = calculate_accuracy(predicted_validation_lists0k7, list(validation_label_dict0.values()))
accuracys1k7 = calculate_accuracy(predicted_validation_lists1k7, list(validation_label_dict1.values()))
accuracys2k7 = calculate_accuracy(predicted_validation_lists2k7, list(validation_label_dict2.values()))

predicted_test_list = []
k = 3
for i in range(940):
  knn_list = find_knn_nearest_list2(k,i)
  predicted_value = find_knn_nearest_answer2(knn_list)
  predicted_test_list.append(predicted_value)

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
test_actual = list(testing_label_dict.values())
print('(W)F1 Score: %.3f' % f1_score(predicted_test_list, test_actual, average='weighted'))
print('(W)Recall: %.3f' % recall_score(predicted_test_list, test_actual, average='weighted'))
print('(W)Precision: %.3f' % precision_score(predicted_test_list, test_actual, average='weighted'))
print('(M)F1 Score: %.3f' % f1_score(predicted_test_list, test_actual, average='macro'))
print('(M)Recall: %.3f' % recall_score(predicted_test_list, test_actual, average='macro'))
print('(M)Precision: %.3f' % precision_score(predicted_test_list, test_actual, average='macro'))