import cv2
import random
import numpy as np
import os
from PIL import Image

Path = r"C:\Users\Markwee\Desktop\CS3244\data\train\8_Negative"
save_path = r"C:\Users\Markwee\Desktop\CS3244\data\train\8_Negative_brightness"

print("Working on it...")
for filename in os.listdir(Path):
    os.chdir(Path)
    full_name = os.path.join(Path, filename)
    img = cv2.imread(full_name)
    def brightness(img, low, high):
        value = random.uniform(low, high)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype = np.float64)
        hsv[:,:,1] = hsv[:,:,1]*value
        hsv[:,:,1][hsv[:,:,1]>255]  = 255
        hsv[:,:,2] = hsv[:,:,2]*value 
        hsv[:,:,2][hsv[:,:,2]>255]  = 255
        hsv = np.array(hsv, dtype = np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img
    img = brightness(img, 0.5, 1.8)
    split = os.path.splitext(filename)
    new_name = split[0] + "_brightness" + split[1]
    #cv2.imshow('Result', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    os.chdir(save_path)
    cv2.imwrite(new_name, img)


