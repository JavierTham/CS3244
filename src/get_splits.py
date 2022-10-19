import os
import pandas as pd
from sklearn.model_selection import train_test_split

labels_path = os.path.join("..", "labels.csv")

df = pd.read_csv(labels_path)
train_df, test_df = train_test_split(df, test_size=0.1, shuffle=True, stratify=df["Species"], random_state=0)
train_df.to_csv(os.path.join("..", "train_split.csv"), index=None)
test_df.to_csv(os.path.join("..", "test_split.csv"), index=None)