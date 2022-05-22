import numpy as np
import os
from matplotlib import pyplot as plt
from keras.models import Model
from sklearn.metrics import *
from utils import *
from keras.models import load_model
from efficientnet import *
import tensorflow as tf
import  keras.backend as K
import time,datetime


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataset = 'IDRID' # IDRID
if dataset =='DDR':
    h, w =  1024, 1024
    image_dir = "./data/DDR/test/images/"
    label_dir = "./data/DDR/test/masks/"
    
if dataset =='IDRID':
    h, w =  960, 1440
    image_dir = "./data/IDRID/test/images/"
    label_dir = "./data/IDRID/test/masks/"

if dataset =='epoch':
    h, w =  960, 1440
    image_dir = "./data/epoch/test/images/"
    label_dir = "./data/epoch/test/masks/"


save_dir='./result/'
model_name='PMCNet.h5'    
images,labels=get_full_test_data(image_dir,label_dir,h,w)
new_labels = make_label(labels)   


print('./weights/'+model_name+'\n')
model=load_model('./'+model_name,compile=False)
probs = model.predict(images, batch_size=1, verbose=1)
save_results(images,probs,labels,save_dir,h,w)

EX = PR_AUC(probs[:,:,:,1],new_labels[:,:,:,1])
HE = PR_AUC(probs[:,:,:,2],new_labels[:,:,:,2])
MA = PR_AUC(probs[:,:,:,3],new_labels[:,:,:,3])
SE = PR_AUC(probs[:,:,:,4],new_labels[:,:,:,4])
print("EX: " +str(EX),"HE: " +str(HE),"MA: " +str(MA),"SE: " +str(SE),'Mean : '+str((EX+HE+MA+SE)/4) )

predictions = np.argmax(probs,axis=-1)

pred = make_label(predictions)   
evaluator_EX= Evaluator()
evaluator_HE= Evaluator()
evaluator_MA= Evaluator()
evaluator_SE= Evaluator()

num = len(pred)
for i in range(num):
    evaluator_EX.update(pred[i,:,:,1], new_labels[i,:,:,1])
    evaluator_HE.update(pred[i,:,:,2], new_labels[i,:,:,2])
    evaluator_MA.update(pred[i,:,:,3], new_labels[i,:,:,3])
    evaluator_SE.update(pred[i,:,:,4], new_labels[i,:,:,4])

ex_dice,ex_iou = evaluator_EX.show()
he_dice,he_iou = evaluator_HE.show()
ma_dice,ma_iou = evaluator_MA.show()
se_dice,se_iou = evaluator_SE.show()
mean_dice = (ex_dice+he_dice + ma_dice + se_dice)/4
mean_iou  = (ex_iou+he_iou + ma_iou + se_iou)/4
print ('mean_dice:',mean_dice, ' mean_iou: ',mean_iou)
print('\n')


