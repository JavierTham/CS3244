import pandas as pd
import os
import shutil

labels = [
    "Chinee apple",
    "Lantana",
    "Parkinsonia",
    "Parthenium",
    "Prickly acacia",
    "Rubber vine",
    "Siam weed",
    "Snake weed",
    "Negative"]

df = pd.read_csv("../labels.csv")

data_folder_path = os.path.join("..", "data")
images_path = os.path.join("..", "data", "images")

# create all the folders
for label in labels:
    os.makedirs(os.path.join(data_folder_path, label))

def move_image(row):
    """
    Takes in a row of the df and moves the image to the folder with its label
    """

    filename = row[0]
    label = row[2]

    src_path = os.path.join(images_path, filename)
    dest_path = os.path.join(data_folder_path, label)
    shutil.move(src_path, dest_path)

if __name__ == "__main__":
    # move all the pictures
    df.apply(lambda row: move_image(row), axis=1)