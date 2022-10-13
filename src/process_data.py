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

from sklearn.model_selection import train_test_split

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

# set paths
data_folder_path = os.path.join("..", "data")
train_folder_path = os.path.join(data_folder_path, "train")
test_folder_path = os.path.join(data_folder_path, "test")
images_path = os.path.join(data_folder_path, "images")
labels_path = os.path.join("..", "labels.csv")

# create directories
os.makedirs(train_folder_path)
os.makedirs(test_folder_path)

# split data into train and test
df = pd.read_csv(labels_path)
train_df, test_df = train_test_split(df, test_size=0.1, shuffle=True, stratify=df["Species"])

# create all the folders for each label in train and test directory
for i in ["train", "test"]:
    for label, idx in labels.items():
        target_dir = os.path.join(data_folder_path, i)
        os.makedirs(os.path.join(target_dir, str(idx)+"_"+label))

def move_image(row, train=False):
    """
    Takes in a row of the df and moves the image to the folder with its label
    
    args:
    row - row of the dataframe
    train - set to True if working on train set 
    """

    filename = row[0]
    label = row[2]
    idx = str(labels[label])

    src_path = os.path.join(images_path, filename)
    dest_path = train_folder_path if train else test_folder_path
    dest_path = os.path.join(dest_path, idx+"_"+label)
    shutil.move(src_path, dest_path)

if __name__ == "__main__":
    # move all the pictures
    train_df.apply(lambda row: move_image(row, train=True), axis=1)
    test_df.apply(lambda row: move_image(row), axis=1)