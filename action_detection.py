#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[2]:


import os
import keras
from keras.regularizers import l2
import argparse
import os
import cv2
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('AGG')
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.layers import (Input, Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D, Input, average,BatchNormalization,LeakyReLU,Conv3D)
from keras.models import Model
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.callbacks import EarlyStopping

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import pickle
import skvideo.io


# ## Loading Data

# In[4]:


input = open("/tf/Activity_detection/data",'rb')
Xin = pickle.load(input)
input1 = open("/tf/Activity_detection/label","rb")
Yin = pickle.load(input1)
print(Xin.shape,Yin.shape)


# In[4]:


class Videoto3D:

    def __init__(self, width, height, depth):
        
        self.width = width
        self.height = height
        self.depth = depth

    def video3d(self, filename,countlis, color=False, skip=True,):
        
        print("coming in video 3d")
        cap = cv2.VideoCapture(filename)
        nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print("number of frame ",nframe)
        if skip:
            frames = [x * nframe / self.depth for x in range(self.depth)]
        else:
            frames = [x for x in range(self.depth)]
        print("frames are",frames)
        #print("size of frame array ",len(frames))
        framearray = []
        f = 0
        for i in range(self.depth):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
            ret, frame = cap.read()
            #print("frame->",frame)
            if frame is None:
              countlis+=1
              # print("coming...........''''''''''''''''''''''''''''in none frame-------->",countlis)
              # print("file is---->",filename)
              continue
            frame = cv2.resize(frame, (self.height, self.width))
            if color:
              framearray.append(frame)
            else:
              framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            
        
        cap.release()
        print("framearray size",np.array(framearray).shape)
        return np.array(framearray)


# In[7]:


def loaddata(video_dir, vid3d, nclass, result_dir,countlis,color=False, skip=True):
    
    
    files = os.listdir(video_dir)
    X = []
    labels = []
    count = 0
    for i in range(nclass):
      path = os.path.join(video_dir, 'c'+str(i),'*.avi')
      files = glob.glob(path)
      print("load data entry point")
      
      for filename in files:
          labels.append(i)
          print("in load data count->",count)
          X.append(vid3d.video3d(filename,countlis,color=False, skip=skip))

    #np.array(x).transpose((0,1,2,3))
    return np.moveaxis(np.array(X),[0,1,2,3],[3,0,1,2]), np.asarray(labels)


# ## Check Video Augmentation for Single Video

# In[5]:


def loadSingleFile(file,vid3d):
    vidarr = []
    vidlabel = []
    vidlabel.append(1)
    count = 0
    vidarr.append(vid3d.video3d(file,count))
    return np.array(vidarr), np.asarray(vidlabel)
img_rows = 120
img_cols = 120
frames = 32
vid3d = Videoto3D(img_rows, img_cols, frames)
file = '/tf/Activity_detection/person01_handwaving_d1_uncomp.avi'
vidarr,vidlabel = loadSingleFile(file,vid3d)
vidout = np.moveaxis(vidarr,[0,1,2,3],[3,0,1,2])
print(vidarr.shape)
print(vidout.shape,type(vidout))
output = vidout.astype(np.uint8)
skvideo.io.vwrite("output_singlevideo.mp4", output)


# ## To do Data Augmentation

# In[6]:


#!apt-get install --no-install-recommends ffmpeg && pip install ffmpeg scikit-video
color = False
skip = True
depth = 32
nmodel = 2

img_rows, img_cols, frames=120, 120, depth
channel=3 if color else 1

nb_classes = nclass
countlis = 0


# In[10]:


from vidaug import augmentors as va
from PIL import Image, ImageSequence
import vidaug.augmentors as va


# In[15]:


vid = vidout[1].astype(np.uint8)
skvideo.io.vwrite("output_one_img.mp4", vid)


# ## Function for Data Augmentation

# In[199]:


sometimes = lambda aug: va.Sometimes(0.5, aug) 
seq = va.Sequential([ 
    sometimes(va.RandomCrop(size=(120, 120))),
    sometimes(va.RandomRotate(degrees=10)),
   sometimes(va.VerticalFlip()),
    sometimes(va.HorizontalFlip()),
    sometimes(va.GaussianBlur(1.5))
])


# In[5]:


X_train, X_test, Y_train, Y_test = train_test_split(Xin, Yin, test_size=0.3, random_state=4)


# In[6]:


X_train = np.expand_dims(X_train,4)
X_test = np.expand_dims(X_test,4)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[177]:


import random
def augmentation(X_train,X_train_aug,Y_train,Y_train_aug,no_of_samples):
    for i in range(no_of_samples):
        idx = random.randint(0,417)
        print("index is ",idx)
        vid = X_train[idx]
        print("shape of vid",vid.shape)
        vid = np.expand_dims(vid,3)
        print("shape of vid after expand",vid.shape)
        video_aug = np.array(seq(vid))
        video_aug = video_aug.squeeze()
        print("video aug shape",video_aug.shape)
        #X_train_aug = np.append(X_train_aug, np.array(video_aug), axis=0)
        X_train_aug.append(video_aug)
        Y_train_aug.append(Y_train[idx])
        #Y_train_aug = np.append(Y_train_aug, np.array(Y_train[idx]),axis=0)
        print(video_aug[0].shape)
    print("len of X_train_aug",len(X_train_aug))
    return X_train_aug,Y_train_aug


# In[79]:


print(X_train.shape)
video_aug = seq(X_train[20])


# In[185]:


skvideo.io.vwrite("aug_vid.mp4",X_train_aug[0])


# In[200]:


X_train_aug = []
Y_train_aug = []
X_train_aug,Y_train_aug = augmentation(X_train,X_train_aug,Y_train,Y_train_aug,120)


# In[55]:


print(train_y.shape)


# In[228]:


#X_train_aug_final = np.array(X_train_aug)
print(type(X_train_aug[0]))
X_train_augmented = np.empty((1,32,120,120))
#print(type(X_train_aug[0][0][0][0]))
count=0
for ele in X_train_aug:
    print(ele.shape)
    ele= np.expand_dims(ele,0)
    print("after expanding",ele.shape)
    X_train_augmented = np.append(X_train_augmented,ele,axis=0)
print(count)


# In[229]:


print(X_train_augmented[1])
X_train_augmented = np.delete(X_train_augmented,0,axis=0)
print(X_train_augmented.shape)


# In[230]:


Y_train_augmented = np.empty((1,6))
for ele in Y_train_aug:
    print(ele.shape)
    ele = np.expand_dims(ele,0)
    print("after expanding",ele.shape)
    Y_train_augmented = np.append(Y_train_augmented,ele,axis=0)
    


# In[231]:


print(Y_train_augmented[1])
Y_train_augmented = np.delete(Y_train_augmented,0,axis=0)
print(Y_train_augmented.shape)


# In[240]:


X_train_augmented = np.expand_dims(X_train_augmented,4)
print(X_train_augmented.shape,Y_train_augmented.shape)


# ## Now concatenating X_train + X_train_augmented && Y_train + Y_train_augmented

# In[241]:


train_X = np.concatenate((X_train, X_train_augmented), axis=0)
train_y = np.concatenate((Y_train, Y_train_augmented), axis=0)
print(train_X.shape)
print(train_y.shape)


# In[7]:


data = open("data_augmented_538","wb")  
pickle.dump(train_X,data)       #dumping X into data pickle for future use.
label = open("label_augmented_538","wb")
pickle.dump(train_y,label) 


# In[273]:


print(os.getcwd())


# ## Import pickle File from system

# In[8]:


input = open("/tf/Activity_detection/data_augmented_538",'rb')
train_X = pickle.load(input)
input1 = open("/tf/Activity_detection/label_augmented_538","rb")
train_y = pickle.load(input1)
print(train_X.shape,train_y.shape)


# ## Create 3D CNN architecture

# In[ ]:





# In[64]:


def Create_3D_CNN(shape,no_of_classes):
    model = Sequential()
    model.add(Dropout(0.20))
    model.add(Conv3D(32,kernel_size=(4,4,32),strides=(1, 1, 1),input_shape=(shape),padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv3D(64,kernel_size=(4,4,32),strides=(1, 1, 1),input_shape=(shape),padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv3D(128,kernel_size=(4,4,32),strides=(1, 1, 1),input_shape=(shape),padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv3D(1,kernel_size=(4,4,32),strides=(1, 1, 1),input_shape=(shape),padding="same"))
    model.add(Activation('sigmoid'))
    model.add(Flatten())
    
    model.add(Dense(no_of_classes, activation='softmax'))
    
    return model


# In[27]:





# In[50]:


shape = (32,120,120,1)
model = Create_3D_CNN((shape), 6)
for layer in model.layers:
    print(layer.get_output_at(0).get_shape().as_list())


# In[51]:


model.summary()


# In[72]:


from keras.optimizers import Adam


# In[10]:


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# In[74]:


import tensorflow as tf
tf.version.VERSION
devices = tf.config.experimental.list_physical_devices("GPU")
print(devices)
devices_names = [d.name.split("e:")[1] for d in devices]
print(devices_names)
n_gpus = 2
strategy = tf.distribute.MirroredStrategy(
           devices=devices_names[:n_gpus])
# learning rate 2*10^-5
with strategy.scope():
    model = Create_3D_CNN((shape), 6)
    opt = Adam(learning_rate=(0.00002*n_gpus))
    model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])


# In[1]:


# model check points 

#model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
history = model.fit(train_X, train_y, validation_data=(X_test, Y_test), batch_size=4, epochs=1, verbose=1,shuffle=True)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

