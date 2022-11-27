# example of zoom image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os
import cv2
import random
import numpy as np
from PIL import Image
# load the images

Path = r"C:\Users\Markwee\Desktop\CS3244\data\train\8_Negative"
save_path = r"C:\Users\Markwee\Desktop\CS3244\data\train\8_Negative_zoom"

print("Working on it...")
for filename in os.listdir(Path):
	os.chdir(Path)
	full_name = os.path.join(Path, filename)
	split = os.path.splitext(filename)
	new_name = split[0] + "_zoom" + split[1]
	img = cv2.imread(full_name)
	# convert to numpy array
	data = img_to_array(img)
	# expand dimension to one sample
	samples = expand_dims(data, 0)
	# create image data augmentation generator
	datagen = ImageDataGenerator(zoom_range=[0.6,0.8])
	# prepare iterator
	it = datagen.flow(samples, batch_size=1)
	# generate samples and plot
	for i in range(9):
		# define subplot
		pyplot.subplot(330 + 1 + i)
		# generate batch of images
		batch = it.next()
		# convert to unsigned integers for viewing
		image = batch[0].astype('uint8')
		# plot raw pixel data
		pyplot.imshow(image)

		os.chdir(save_path)
		cv2.imwrite(new_name, image)
	# show the figure
	#pyplot.show()