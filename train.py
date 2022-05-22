import os
import cv2
import random
import keras
import numpy as np 
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import keras.backend as K
from models import *
from utils import *
import tensorflow as tf

def multi_class(y_true, y_pred):
    CE = K.categorical_crossentropy(y_true, y_pred)
    return  CE   

def step_decay(epoch):
    initial_lrate = 0.0001
    epochs_drop = 60
    lrate = initial_lrate * math.pow(1-(1+epoch)/epochs_drop,0.9)
    return lrate

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   
sess = tf.Session(config=config)
K.set_session(sess)

dataset = 'DDR' 
if dataset =='DDR':
    image_h, image_w =  1024, 1024
    train_path = "./data/DDR/train"
    valid_path = "./data/DDR/valid"
    
if dataset =='IDRID':
    image_h, image_w =  960, 1440
    train_path = "./data/IDRID/train_aug"
    valid_path = "./data/IDRID/valid"
    
if dataset =='epoch':
    image_h, image_w =  960, 1440
    train_path = "./data/epoch/train"
    valid_path = "./data/epoch/valid"

batch_size = 1
train_num = len(os.listdir(train_path+'/images/'))
valid_num  = len(os.listdir(valid_path+'/images/'))

lr_decay = keras.callbacks.LearningRateScheduler(step_decay,verbose=1)
checkpointer=ModelCheckpoint(filepath='./weights/PMCNet_DDR{epoch:02d}.h5',verbose=1,monitor='val_acc')
loss_fun= multi_class
model = PMCNet (image_h, image_w, color_type=3, num_class=5) 
model.compile(optimizer = Adam(lr = 0.0001), loss = loss_fun, metrics = ['acc'])  
                                                                                   
train_data=dataGenerator(batch_size=batch_size,target_size=(image_h, image_w),train_path=train_path)

valid_data=dataGenerator(batch_size=batch_size,target_size=(image_h, image_w),train_path=valid_path)

model.fit_generator(train_data,steps_per_epoch=train_num/batch_size,epochs=60,validation_data=valid_data,
                    workers=1,validation_steps=valid_num/batch_size,callbacks=[checkpointer,lr_decay])

