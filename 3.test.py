# Importing the libraries
import cv2
import numpy as np 
from keras.models import load_model
import argparse
from PIL import Image
import imutils
import matplotlib.pyplot as plt

# Defining the loss function
def mean_squared_loss(x1,x2):
    difference=x1-x2
    a,b,c,d,e=difference.shape
    n_samples=a*b*c*d*e
    sq_difference=difference**2
    Sum=sq_difference.sum()
    distance=np.sqrt(Sum)
    mean_distance=distance/n_samples
    return mean_distance

# Loading the model
model=load_model("./model_files/saved_model.h5")

# Loading the video and predicting each frames
cap = cv2.VideoCapture("TESTING.mp4")
count = 0
while cap.isOpened():
    imagedump=[]
    ret,frame=cap.read()
    for i in range(10):
        ret,frame=cap.read()
        image = imutils.resize(frame,width=700,height=600)
        frame=cv2.resize(frame, (227,227), interpolation = cv2.INTER_AREA)
        gray=0.2989*frame[:,:,0]+0.5870*frame[:,:,1]+0.1140*frame[:,:,2]
        gray=(gray-gray.mean())/gray.std()
        gray=np.clip(gray,0,1)
        imagedump.append(gray)
    imagedump=np.array(imagedump)
    imagedump.resize(227,227,10)
    imagedump=np.expand_dims(imagedump,axis=0)
    imagedump=np.expand_dims(imagedump,axis=4)
    output=model.predict(imagedump)
    loss=mean_squared_loss(imagedump,output)
    if frame.any()==None:
        print("none")
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
    if loss>0.00068:
        print('Abnormal Event Detected')
        # Save the video if any anomaly is detected
        cv2.imwrite("./anomal_images/"+str(count)+".jpg", image)
        cv2.putText(image,"Abnormal Event",(100,80),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
    cv2.imshow("video",image)
    count += 1
cap.release()
cv2.destroyAllWindows()