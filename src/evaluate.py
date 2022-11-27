import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import cv2
import shutil
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn

##Load Data
print("Loading Data")



##############################################################################################
"""
Remarks: 1. Put this evaluate script inside a train folder, inside train folder there should be test_data, train_data.
            
         2. Specify the absolute path of the weights to be evaluated.
         
         3. Specify the result folder name
         
         4. Select GPU: 0 or 1
            
"""

img_width        = 256
img_height       = 256

model_path       = r"C:\Users\Markwee\Desktop\CS3244_proj\dataset\M1_new.h5"

epoch_num = model_path.split("\\")[-1].split(".")[1].split('-')[0]
result_name      = epoch_num

model_name       = model_path.split("\\")[-3]

train_folder_dir = os.getcwd()
result_save_path = os.path.join(train_folder_dir ,model_name,'result',result_name)
DATA_DIR          = os.path.join(train_folder_dir ,'test')
#DATA_DIR         = os.path.join(train_folder_dir ,'test_data (from TrainFolder_9)')

# Select GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


##############################################################################################

if not os.path.exists(result_save_path):
  os.makedirs(result_save_path)

shutil.copy(model_path,result_save_path)


train_count = []
data = []
y_true = []



CLASSES = os.listdir(DATA_DIR)
print(CLASSES)


for class_name in os.listdir(DATA_DIR):
    print(class_name)
    for img_path in os.listdir(os.path.join(DATA_DIR, class_name)):
        y_true.append(CLASSES.index(class_name))
        img = cv2.imread(os.path.join(DATA_DIR,class_name, img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_width,img_height))
        img = img/255.
        
        data.append(img)
    
    for img_path in os.listdir(os.path.join(DATA_DIR, class_name)):
        train_count.append(CLASSES.index(class_name))
    
arr_ = np.array(data)

y_true = np.array(y_true)
arr = arr_.reshape(-1, img_height,img_width,3)
print(arr.shape)

##Load Model
print("Loading Model")

model = load_model(model_path)
pred = model.predict(arr, verbose=1)
pred = np.argmax(pred,axis=1)


for i,(p,y) in enumerate(zip(pred, y_true)):
    curr_img = cv2.cvtColor(((arr_[i])*255).astype('uint8'), cv2.COLOR_RGB2BGR)
    folder_name = str(y)+str(p)
    folder_path = os.path.join(result_save_path,folder_name)
    if(not os.path.exists(folder_path)):
        os.mkdir(folder_path)
    img_name = str(len(os.listdir(folder_path)))+'.jpg'
    cv2.imwrite(os.path.join(folder_path, img_name), curr_img)

print('Classfication report')
target_names = ['0_Chinee apple', '1_Lantana', '2_Parkinsonia', '3_Parthenium', '4_Prickly acacia', '5_Rubber vine', '6_Siam weed', '7_Snake weed', '8_Negative']
#print(classification_report(y_true, pred, target_names=target_names))

clsf_report = pd.DataFrame(classification_report(y_true, pred, target_names = target_names,output_dict=True)).transpose()
clsf_report.to_csv(os.path.join(result_save_path, 'class_report.csv'), index= True)

print('Confusion Matrix')
cm = confusion_matrix(y_true, pred)
print(cm)

class_list = CLASSES
class_list_w_spaces = [ i+ ' '*8 for i in class_list ]  #class_list_w_spaces = ["        General", "     LIFE", "        STA_Green", "       White", "       Without PPE"]
df_cm = pd.DataFrame(cm, class_list_w_spaces, class_list)
plt.figure(figsize=(10,10))
sns_plot = sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, fmt='g') # font size
sns_plot.set(xlabel='Prediction', ylabel='Ground Truth')
sns_plot.tick_params(labelsize=6)

sns_plot.set(title = "M1_original_256")

sns_plot.figure.savefig(os.path.join(result_save_path, 'confusion_matrix')+'.png')
plt.show()

"""
train_count_string = "({}, {}, {}, {})".format(train_count.count(0),train_count.count(1),train_count.count(2),train_count.count(3))
input_dim_string = "({}, {})".format(IMG_SIZE, IMG_SIZE)
precision_string = "({:.2f},{:.2f},{:.2f},{:.2f})".format(float(c_report['General']['precision']),float(c_report['LIFE']['precision']),float(c_report['STA_Green']['precision']),float(c_report['Without PPE']['precision']))
recall_string = "({:.2f},{:.2f},{:.2f},{:.2f})".format(float(c_report['General']['recall']),float(c_report['LIFE']['recall']),float(c_report['STA_Green']['recall']),float(c_report['Without PPE']['recall']))
f1_score_string = "({:.2f},{:.2f},{:.2f},{:.2f})".format(float(c_report['General']['f1-score']),float(c_report['LIFE']['f1-score']),float(c_report['STA_Green']['f1-score']),float(c_report['Without PPE']['f1-score']))
support_string = "({}, {}, {}, {})".format(int(c_report['General']['support']),int(c_report['LIFE']['support']),int(c_report['STA_Green']['support']),int(c_report['Without PPE']['support']))
macro_avg_string = "({:.2f},{:.2f},{:.2f},{})".format(float(c_report['macro avg']['precision']), float(c_report['macro avg']['recall']), float(c_report['macro avg']['f1-score']), int(c_report['macro avg']['support']))
accuracy_string = "{:.2f}".format(float(c_report['accuracy']))
cm_string = "{}:{}:{}:{}".format(cm[0,:],cm[1,:],cm[2,:],cm[3,:])
print("{}:{}:{}:{}:{}:{}:{}:{}:{}:{}".format("ResNet50", train_count_string, input_dim_string,precision_string, recall_string, f1_score_string, support_string, macro_avg_string, accuracy_string, cm_string))
"""