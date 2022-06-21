import cv2
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Y','Z']
n_classes = len(classes)

x_train,x_test,y_train,y_test = train_test_split(X,y, test_size = 2500, train_size = 7500, random_state = 9)

x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0
clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(x_train_scaled, y_train)

y_predict = clf.predict(x_test_scaled)
accuracy = accuracy_score(y_test,y_predict)
print("accuracy = ",accuracy)

cap = cv2.VideoCapture(0)

while(True):
    try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLORBGR2GRAY)
        height,width = gray.shape
        upper_left = (int(width/2-56),int(height/2-56))
        bottom_right = (int(width/2+56),int(height/2+56))

        #creating rec & roi
        cv2.rectangle(gray,upper_left,bottom_right,(0,255,0),2)
        roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
        im_pil = Image.fromarray(roi)
        image_bw = im_pil.convert('L')
        image_bw_resized = image_bw.resize((28,28),Image.ANTIALIAS)

        #inverting image
        img_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)

        #converting to scaler quantity 
        pixel_filter = 20
        min_pixel = np.percentile(img_bw_resized_inverted,pixel_filter)

        #scaling/resticting
        img_bw_resized_inverted_scaled = np.clip(img_bw_resized_inverted-min_pixel,0,255)

        max_pixel = np.max(img_bw_resized_inverted)

        #converting to array
        img_bw_resized_inverted_scaled = np.asarray(img_bw_resized_inverted_scaled)/max_pixel

        #test sample & prediction
        test_sample = np.array(img_bw_resized_inverted_scaled).reshape(1,784)
        test_predict = clf.predict(test_sample)
        print(test_predict)

        #displaying resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('key'):
            break

    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()