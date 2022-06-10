# Importing Libraries
import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from datetime import timedelta
from IPython.display import clear_output
from keras.preprocessing.image import img_to_array,load_img

from utility.time import format_timedelta
from utility.duration import get_saving_frames_durations

# Defining the hyper-parameters
SAVING_FRAMES_PER_SECOND = 5
# Directory of the training videos
vid_dir = "./Dataset/Avenue_Dataset/training_videos/"
count = 0

# Extracting frame from videos
videos = os.listdir(vid_dir)
for video in videos:
    if video.split(".")[1]=="avi":
        video_file = vid_dir + video
        filename, _ = os.path.splitext(video_file)
        print(filename)
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
        saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
        while True:
            is_read, frame = cap.read()
            if not is_read:
                break
            frame_duration = count / fps
            try:
                closest_duration = saving_frames_durations[0]
            except IndexError:
                break
            if frame_duration >= closest_duration:
                frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
                cv2.imwrite(os.path.join(r"C:\Users\siyam\Desktop\Github\minavera\train\frames", f"frame{count}.jpg"), frame)
                try:
                    saving_frames_durations.pop(0)
                except IndexError:
                    pass
            # increment the frame count
            count += 1

# Directory of training images (Combined two dataset)
images_path = "./train/frames/"
images_list = os.listdir(images_path)

# Spilitting the image list into 4 chunks to utilize multithreading
import numpy
four_image_list = numpy.array_split(images_list,4);

first_image_list = four_image_list[0]
second_image_list = four_image_list[1]
third_image_list = four_image_list[2]
fourth_image_list = four_image_list[3]

# Defining the multi-threads
from threading import *

class Part_1(Thread):
    
    def __init__(self, image_list):
        super(Part_1, self).__init__()
        self.image_list = images_list
    def run(self):
        store_image=[]
            
        for i in tqdm(range(len(images_list))):
            full_path = images_path+images_list[i]
            image=load_img(full_path)
            image=img_to_array(image)
            image=cv2.resize(image, (227,227), interpolation = cv2.INTER_AREA)
            gray=0.2989*image[:,:,0]+0.5870*image[:,:,1]+0.1140*image[:,:,2]
            store_image.append(gray)
            
        store_image=np.array(store_image)
        a,b,c=store_image.shape
        store_image.resize(b,c,a)
        store_image=(store_image-store_image.mean())/(store_image.std())
        store_image=np.clip(store_image,0,1)
        np.save("./model_files/training_first.npy",store_image)

class Part_2(Thread):
    
    def __init__(self, images_list):
        super(Part_2, self).__init__()
        self.images_list = images_list
    
    def run(self):
        store_image=[]
            
        for i in (range(len(images_list))):
            full_path = images_path+images_list[i]
            image=load_img(full_path)
            image=img_to_array(image)
            image=cv2.resize(image, (227,227), interpolation = cv2.INTER_AREA)
            gray=0.2989*image[:,:,0]+0.5870*image[:,:,1]+0.1140*image[:,:,2]
            store_image.append(gray)
            
        store_image=np.array(store_image)
        a,b,c=store_image.shape
        store_image.resize(b,c,a)
        store_image=(store_image-store_image.mean())/(store_image.std())
        store_image=np.clip(store_image,0,1)
        np.save("./model_files/training_second.npy",store_image)
        
class Part_3(Thread):
    
    def __init__(self, images_list):
        super(Part_3, self).__init__()
        self.images_list = images_list
    
    def run(self):
        store_image=[]
        
        for i in (range(len(images_list))):
            full_path = images_path+images_list[i]
            image=load_img(full_path)
            image=img_to_array(image)
            image=cv2.resize(image, (227,227), interpolation = cv2.INTER_AREA)
            gray=0.2989*image[:,:,0]+0.5870*image[:,:,1]+0.1140*image[:,:,2]
            store_image.append(gray)

        store_image=np.array(store_image)
        a,b,c=store_image.shape
        store_image.resize(b,c,a)
        store_image=(store_image-store_image.mean())/(store_image.std())
        store_image=np.clip(store_image,0,1)
        np.save("./model_files/training_third.npy",store_image)
        
class Part_4(Thread):
    
    def __init__(self, images_list):
        super(Part_4, self).__init__()
        self.images_list = images_list
    
    def run(self):
        store_image=[]
        
        for i in (range(len(images_list))):
            full_path = images_path+images_list[i]
            image=load_img(full_path)
            image=img_to_array(image)
            image=cv2.resize(image, (227,227), interpolation = cv2.INTER_AREA)
            gray=0.2989*image[:,:,0]+0.5870*image[:,:,1]+0.1140*image[:,:,2]
            store_image.append(gray)
            
        store_image=np.array(store_image)
        a,b,c=store_image.shape
        store_image.resize(b,c,a)
        store_image=(store_image-store_image.mean())/(store_image.std())
        store_image=np.clip(store_image,0,1)
        np.save("./model_files/training_fourth.npy",store_image)


t1 = Part_1(first_image_list)
t2 = Part_2(second_image_list)
t3 = Part_3(third_image_list)
t4 = Part_4(fourth_image_list)

# Starting the threads
t1.start()
t2.start()
t3.start()
t4.start()

# Merging the numpy files 
import numpy as np
training_data1=np.load('./model_files/training_first.npy')
training_data2=np.load('./model_files/training_second.npy')
training_data3=np.load('./model_files/training_third.npy')
training_data4=np.load('./model_files/training_fourth.npy')

merged = [*first_numpy, *second_numpy, *third_numpy, *fourth_numpy]

np.save('./model_files/training.npy.npy',merged)

