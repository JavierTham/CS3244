import shutil
import random
import glob

#data_folder_path = r"C:\Users\Markwee\Desktop\CS3244\data\train\8_Negative"
train = r"C:\Users\Markwee\Desktop\CS3244\data\trainneg" #seperate folder for 1000 chosen negative train images
test = r"C:\Users\Markwee\Desktop\CS3244\data\testneg" #seperate folder for 100 chosen negative test images
random.seed(1)

tbm_train = random.sample(glob.glob("C:/Users/Markwee/Desktop/CS3244/data/train/8_Negative/*.png"), 1000) #set path to initial 8195 chosen negative train images
tbm_test = random.sample(glob.glob("C:/Users/Markwee/Desktop/CS3244/data/test/8_Negative/*.png"), 100) #set path to initial chosen negative test images

for f in enumerate(tbm_train, 1):
	shutil.copy(f[1], train)

for f in enumerate(tbm_test, 1):
	shutil.copy(f[1], test)

#after running, you can add these folders back into your train and test folders with the other classes