import os
import cv2
import math
import random
import keras
import numpy as np
import keras.backend as K
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16

from utils import *
from efficientnet import *
from efficientnet.layers import Swish

def conv_unit(inputs, nb_filter):
    kernel = 3
    x = Conv2D(nb_filter, (kernel, kernel), strides=1, padding='same')(inputs)
    #x = BatchNormalization() (x)
    x = LeakyReLU(alpha=0.0)  (x)
    return x

def conv_block(inputs, nb_filter):
    x = conv_unit(inputs, nb_filter)
    x = conv_unit(x, nb_filter)
    return x

def Upsample_conv(inputs, nb_filter):
    x = UpSampling2D(size=(2, 2),interpolation='bilinear')(inputs)
    x = Conv2D(nb_filter, (1, 1), strides=1, padding='same')(x)   
    x = LeakyReLU(alpha=0.0)  (x)
    return x

def skip_block(de,en):
    concat = add([en, de])
    #concat = concatenate([en, de],axis=-1)
    return concat
    
def DAB_block(concat):  ## CA+SA+PA
    # CA
    shape = K.int_shape(concat)
    x = AveragePooling2D(pool_size=(shape[1], shape[2]), padding='same') (concat)
    x = Dense(shape[3]) (x)
    score_c = Activation('sigmoid') (x)
    CA = Multiply() ([concat,score_c])
    
    # SA
    s_avg = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True)) (concat)
    score_s = Activation('sigmoid') (s_avg)
    SA = Multiply() ([concat,score_s])
    
    # PA
    x =  Conv2D(shape[3], (1, 1), strides = 1, padding = 'same')(concat)
    score_p = Activation('sigmoid') (x)
    PA = Multiply() ([concat,score_p])
    
    out = add ([CA,SA,PA])
    return out

def PFF_3(pre,cur,nex):
    shape = K.int_shape(cur)
    
    down = DepthwiseConv2D(2, strides = 2, padding = 'same') (pre)
    down =  Conv2D(shape[3], (1, 1), strides = 1, padding = 'same')(down)
    
    up =  Conv2D(shape[3], (1, 1), strides = 1, padding = 'same')(nex)
    up = UpSampling2D(size=(2, 2),interpolation='bilinear')(up)
    
    mul_low = Multiply() ([down,cur])
    mul_high = Multiply() ([up,cur])
    
    concat = concatenate([mul_low,mul_high], axis = -1)
    
    att = DAB_block(concat)
    x =  Conv2D(shape[3], (1, 1), strides = 1, padding = 'same')(att)
    x = LeakyReLU(alpha=0.0)  (x)
    return x
    
def PFF_2_pre(cur,nex):
    shape = K.int_shape(cur)
    up =  Conv2D(shape[3], (1, 1), strides = 1, padding = 'same')(nex)
    up = UpSampling2D(size=(2, 2),interpolation='bilinear')(up)
    mul = Multiply() ([cur,up])
   
    concat = concatenate([cur,mul], axis = -1)
    att = DAB_block(concat)
    x =  Conv2D(shape[3], (1, 1), strides = 1, padding = 'same')(att)
    x = LeakyReLU(alpha=0.0)  (x)
    
    return x 

def PFF_2_nex(pre,cur):
    shape = K.int_shape(cur)
    down = DepthwiseConv2D(2, strides = 2, padding = 'same') (pre)
    down =  Conv2D(shape[3], (1, 1), strides = 1, padding = 'same')(down)
    mul = Multiply() ([down,cur])

    concat = concatenate([cur,mul], axis = -1)
    att = DAB_block(concat)
    x =  Conv2D(shape[3], (1, 1), strides = 1, padding = 'same')(att)
    x = LeakyReLU(alpha=0.0)  (x)
    return x
    
def PMCNet(img_rows, img_cols, color_type, num_class):
    base_model = EfficientNetB0(include_top = False, weights = None, input_shape = (img_rows, img_cols, color_type))
    base_model.load_weights("./weights/EfficientNetB0.h5")
    nb_filter = [32,96,144,240,672]  
    conv2_1 = base_model.get_layer("swish_4").output 
    conv3_1 = base_model.get_layer("swish_10").output
    conv4_1 = base_model.get_layer("swish_16").output 
    conv5_1 = base_model.get_layer("swish_34").output
    
    conv5_1_Att_PFF = PFF_2_nex(conv4_1,conv5_1)
    conv4_1_Att_PFF = PFF_3 (conv3_1,conv4_1,conv5_1)
    conv3_1_Att_PFF = PFF_3 (conv2_1,conv3_1,conv4_1)
    conv2_1_Att_PFF = PFF_2_pre(conv2_1,conv3_1)
    
    conv2_1_Att_PFF = add([conv2_1_Att_PFF,conv2_1])
    conv3_1_Att_PFF = add([conv3_1_Att_PFF,conv3_1])
    conv4_1_Att_PFF = add([conv4_1_Att_PFF,conv4_1])
    conv5_1_Att_PFF = add([conv5_1_Att_PFF,conv5_1])
    
    de4 = Upsample_conv(conv5_1_Att_PFF, nb_filter[3])
    concat4 = skip_block(de4, conv4_1_Att_PFF)
    concat4 = conv_block(concat4, nb_filter[3])
    
    de3 = Upsample_conv(concat4, nb_filter[2])
    concat3 = skip_block(de3, conv3_1_Att_PFF)
    concat3 = conv_block(concat3, nb_filter[2])
    
    de2 = Upsample_conv(concat3, nb_filter[1])
    concat2 = skip_block(de2, conv2_1_Att_PFF)
    concat2 = conv_block(concat2, nb_filter[1])
        
    out = Upsample_conv(concat2, nb_filter[0])
    out = conv_block(out, nb_filter[0])
    out = Conv2D(num_class, (1, 1), activation='softmax', padding='same')(out)   
    model = Model(base_model.input,out)
    model.summary()
    return model
  
    
