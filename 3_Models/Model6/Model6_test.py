''' Model 6 trains a 2-branch CNN using the original tiles, road tiles, and building tiles.'''
''' It outputs a .csv file with the actual and predicted Ys after each loop for later analysis'''

## Imports

import tensorflow as tf
from tensorflow import math as math
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, InputLayer
import torch
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd

#For image manipulation and resizing
import cv2
import PIL
from PIL import Image
from PIL import Image, ImageOps

#For file navigation
import os
import glob
import sklearn
#For Plotting VGG-16 architecture model

#VGG model import
from tensorflow.keras.applications.vgg16 import VGG16

from util import *
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.python.client import device_lib
import tensorflow_addons as tfa
import time
import sys

#---Setting up Run---#
#Getting loop information
j = sys.argv[1]
#Resetting and setting up GPU for model run.
reset_keras2()

##---Data Prep---##
#Reading Path
count_path='./population_count_final.csv'
tiles_path1='./tiles/'
tiles_path2='./tiles/{}'
building_tiles1='./building_tiles/'
building_tiles2='./building_tiles/{}'
road_tiles1='./road_tiles/'
road_tiles2='./road_tiles/{}'

#Reading pop count data in
df=pd.read_csv(count_path)

#Making list of image paths
tile_images= [f for f in os.listdir(tiles_path1) if f.endswith('.tif')]
building_images= [f for f in os.listdir(building_tiles1) if f.endswith('.tif')]
road_images= [f for f in os.listdir(road_tiles1) if f.endswith('.tif')]
sort_nicely(tile_images) #Sorting image names in order
sort_nicely(building_images)
sort_nicely(road_images)

#Stacking images in a list
image_stack1=[]

for tile in tile_images:
    im=Image.open(tiles_path2.format(tile)) #Open tile
    im=im.resize([224,224]) #Resize tile to 224x224
    image_stack1.append(im)

image_stack2=[]

for tile in building_images:
    im=Image.open(building_tiles2.format(tile)) #Open tile
    im=im.resize([224,224]) #Resize tile to 224x224
    image_stack2.append(im)

image_stack3=[]

for tile in road_images:
    im=Image.open(road_tiles2.format(tile)) #Open tile
    im=im.resize([224,224]) #Resize tile to 224x224
    image_stack3.append(im)

#Cropping data to only include Boane data + data for 16 zero tiles.
X = image_stack1[:140]
X_building = image_stack2[:140]
X_road = image_stack3[:140]
Y = df['pop'].tolist()[:140]

#Combining road and building binary maps into one map.
X2=combine(X_building,X_road)

#Splitting dataset
#Seed is to ensure data splits match b/w original tiles and other tiles.
seed=np.random.randint(low=0,high=999999,size=1)[0]
X_train, X_test, Y_train, Y_test = train_test_split1(X,Y,0.15,seed)
X_train2, X_test2, Y_train2, Y_test2 = train_test_split1(X2,Y,0.15,seed)

seed=np.random.randint(low=0,high=999999,size=1)[0]
X_train, X_val, Y_train, Y_val = train_test_split1(X_train,Y_train,0.15,seed)
X_train2, X_val2, Y_train2, Y_val2 = train_test_split1(X_train2,Y_train2,0.15,seed)

#Augmenting data & turning images to numpy arrays
#Note that val and test sets do not actually get augmented.
X_train,Y_train=Augment(X_train,Y_train)
X_train2,Y_train2=Augment(X_train2,Y_train2)

X_val,Y_val=Augment(X_val,Y_val)
X_val2,Y_val2=Augment(X_val2,Y_val2)

X_test,Y_test=Augment(X_test,Y_test)
X_test2,Y_test2=Augment(X_test2,Y_test2)

#Standardizing images (two options in function: 0 to 1 and mean/std normalizing)
X_train, X_val, X_test=standardize(X_train,X_val,X_test) #Normalize based on Imagenet mean/std
X_train2, X_val2, X_test2=standardize2(X_train2,X_val2,X_test2) #Standardize 0 to 1

#Converting X & Ys to tensor
X_train=tf.convert_to_tensor(X_train)
X_val=tf.convert_to_tensor(X_val)
X_test=tf.convert_to_tensor(X_test)

X_train2=tf.convert_to_tensor(X_train2)
X_val2=tf.convert_to_tensor(X_val2)
X_test2=tf.convert_to_tensor(X_test2)

Y_train=tf.convert_to_tensor(Y_train)
Y_val=tf.convert_to_tensor(Y_val)
Y_test=tf.convert_to_tensor(Y_test)

##--- MODEL CREATION & TRAINING ---##

#Creating Model
#2-branch CNN model. One branch is pre-trained with ImageNet, other is randomly intialized.
#RGB-branch of CNN is frozen.
cnn_model=concat_model2()

#Defining optimizer, loss function, and metrics for trainer

#Defining optimizer for first run
#Two options: set 2e-3 or a learning schedule with decay

#Option 1
opt1= tf.keras.optimizers.Adam(learning_rate=2e-2)

#Option 2
'''
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=8,
    decay_rate=0.5)
opt1= tf.keras.optimizers.Adam(lr_schedule)
'''

#Defining optimizer for 2nd run when all layers unfrozen
#Discriminative training defined below.

#Defining exponential taper of learning rate.
learning_rates=[]

for i in range(5):
    lr= (2e-3)*(((1/10)**(1/10))**(i))
    lr= tf.keras.optimizers.Adam(learning_rate=lr)
    learning_rates.append(lr)

for i in range((len(cnn_model.layers)-5)//2):
    lr= (2e-3)*(((1/10)**(1/10))**(i+5))
    lr= tf.keras.optimizers.Adam(learning_rate=lr)
    learning_rates.append(lr)
    learning_rates.append(lr)

learning_rates=list(reversed(learning_rates))
optimizers_and_layers=[]

#Assigning learning rates to layers.
for i in range(len(cnn_model.layers)):
    optimizers_and_layers.append((learning_rates[i],cnn_model.layers[i]))

#Setting up optimizer with discriminative training
opt2 = tfa.optimizers.MultiOptimizer(optimizers_and_layers)

#Defining EarlyStopping
es=[EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=10, restore_best_weights=True)]

##---First Run

#Train last layer
cnn_model.compile(optimizer=opt1, loss='mean_absolute_error', metrics=['mean_absolute_error'])
cnn_model.fit(x=[X_train,X_train2],y=Y_train, batch_size=32, epochs=50, verbose=1, validation_data=([X_val,X_val2],Y_val),callbacks=[es])

#---Second Run
#Unfreeze all layers
for layer in cnn_model.layers:
    layer.trainable=True

#Train all layers
cnn_model.compile(optimizer=opt2, loss='mean_absolute_error', metrics=['mean_absolute_error'])
history=cnn_model.fit(x=[X_train,X_train2],y=Y_train, batch_size=32, epochs=50, verbose=1, validation_data=([X_val,X_val2],Y_val),callbacks=[es])

#Predicting Test Data & Saving to CSV
start_time = time.time()
Y_pred=cnn_model.predict([X_test,X_test2])
end_time = time.time()
print("-------- Prediction took {}s".format(end_time-start_time))
final_df=pd.DataFrame(np.zeros([len(Y_test),2]),columns=['actual','predicted'])
final_df['actual']=Y_test
final_df['predicted']=Y_pred

final_df.to_csv('results/model6/model6_df{}.csv'.format(j),index=False)
print('Loop {} completed'.format(j))
tf.keras.backend.clear_session()
