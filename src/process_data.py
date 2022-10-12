##########
# README #
##########

# 1. download the images from the original github https://github.com/AlexOlsen/DeepWeeds
# 2. in the github repo, go into labels and download/copy the labels.csv file
# 3. organise your folders as shown
# ├── data
# │   └── images
# │       ├── 20180322-133901-1.jpg
# |       ├── ...
# │       └── ______.jpg
# ├── src
# │   └── process_data.py
# └── labels.csv
# 4. run the python code from inside src directory, `python process_data.py`  

import pandas as pd
import os
import shutil

labels = dict([
    ("Chinee apple", 0),
    ("Lantana", 1),
    ("Parkinsonia", 2),
    ("Parthenium", 3),
    ("Prickly acacia", 4),
    ("Rubber vine", 5),
    ("Siam weed", 6),
    ("Snake weed", 7),
    ("Negative", 8)
])

df = pd.read_csv(os.path.join("..", "labels.csv"))

data_folder_path = os.path.join("..", "data")
images_path = os.path.join("..", "data", "images")

# create all the folders
for label, idx in labels.items():
    os.makedirs(os.path.join(data_folder_path, str(idx)+"_"+label))

def move_image(row):
    """
    Takes in a row of the df and moves the image to the folder with its label
    """

    filename = row[0]
    label = row[2]
    idx = str(labels[label])

    src_path = os.path.join(images_path, filename)
    dest_path = os.path.join(data_folder_path, idx+"_"+label)
    shutil.move(src_path, dest_path)

if __name__ == "__main__":
    # move all the pictures
    df.apply(lambda row: move_image(row), axis=1)