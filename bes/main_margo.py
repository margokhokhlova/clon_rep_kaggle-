#!/usr/bin/env python
# coding: utf-8

# In[17]:


from keras import backend as K
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from models.models import get_model
from losses import make_loss
import os
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import numpy as np
from matplotlib.image import imread
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.metrics import  binary_accuracy


# In[15]:


from dataloaders.inria_datagenerator import InriaDatagenerator
from segmentation_models.backbones import get_preprocessing as process_image
from dataloaders.Losses import my_iou_metric as iou_score


# In[ ]:


# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0"; 
config = tf.ConfigProto()
session = tf.Session(config=config)
K.set_session(session)


# In[4]:


BACKBONE = 'resnext50'
preprocess_input = process_image(BACKBONE)

# load original dataset
import glob
X_train = sorted(glob.glob("/data/margokat/inria/clean_data/train_frames/*.png"))
y_train = sorted(glob.glob("/data/margokat/inria/clean_data/train_masks/*.png"))
X_val = sorted(glob.glob("/data/margokat/inria/clean_data/val_frames/*.png"))
y_val = sorted(glob.glob("/data/margokat/inria/clean_data/val_masks/*.png"))
print('The number of training samples is =  %d, validation samples is = %d' % (len(X_train), len(X_val)))


# calculate coverage
coverage_train = {}
gr_truth = np.zeros((len(y_train ),1))
for i in range(len(y_train )):
    mask = imread(y_train [i])
    H, W = mask.shape
    gr_truth[i] =  np.sum(mask) / (H*W)
    coverage_train[y_train[i]] = gr_truth[i]

coverage_val = {}
gr_truth = np.zeros((len(y_val ),1))
for i in range(len(y_val )):
    mask = imread(y_val[i])
    H, W = mask.shape
    gr_truth[i] =  np.sum(mask) / (H*W)
    coverage_val[y_val[i]] = gr_truth[i]


# In[15]:


# define parameters for sampling

params = {'dim': (512, 512),
          'batch_size': 5,
          'n_channels_img':3,
          'n_channel_mask':1,
          'shuffle': False,
          'Flip': True,
          'aux_loss': True}

training_generator = InriaDatagenerator(X_train, y_train, Transform = True, stratified_sampling=False, coverage=coverage_train, Process_function = preprocess_input, **params)

params = {'dim': (512, 512),
          'batch_size': 10,
          'n_channels_img':3,
          'n_channel_mask':1,
          'shuffle': False,
          'aux_loss': True}

validation_generator = InriaDatagenerator(X_val, y_val, Transform = True, stratified_sampling=True, coverage=coverage_val, Process_function = preprocess_input, **params)
(X,y) = training_generator.__getitem__(500, processing = False)
#for i in range(X.shape[0]):
#    show_sample(X[i,:].astype(int), np.squeeze(y[i,:]))


# In[8]:


def sig_iou_score(y_true, y_pred):
    return iou_score(y_true,tf.math.sigmoid(y_pred)) # tf.math.sigmoid(y_pred)

def sigm_binary_accuracy(y_true, y_pred):
    return binary_accuracy(y_true, tf.math.sigmoid(y_pred))  #tf.math.sigmoid(y_pred)


loss_function ='seloss'

# Get the model
model = get_model(network = 'unet_resnext_50_margo',input_shape=(512, 512, 3),
                                      freeze_encoder=False)
Adam_opt =Adam(lr=0.00005,  beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)#, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=True)


# In[ ]:



loss_history = []
weight_path = "/data/margokat/models_saved/inria/{}_weights.best.hdf5".format('resnext50_unet_margo_se')
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only=True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2, verbose=1, mode='auto',
                                   epsilon=0.0001, cooldown=4, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss",
                      mode="min",
                      patience=15)  # probably needs to be more patient, but kaggle time is limited

tboard = TensorBoard(log_dir='Graph_resnext50_unet_512_modif_transpose/margo_se/', histogram_freq=0,
          write_graph=True, write_images=True)

callbacks_list = [checkpoint, early, tboard]

# fit the model
print('Training the model...')
model.compile(Adam_opt,loss={'prediction': 'binary_crossentropy',
                             'se_auxillary': 'binary_crossentropy'},loss_weights={'prediction': 1.0,
                            'se_auxillary': 1.0}, metrics=[sig_iou_score, sigm_binary_accuracy])
loss_history = model.fit_generator(generator=training_generator, steps_per_epoch=np.ceil(len(X_train)/params['batch_size']),
        epochs=120,
        validation_data=validation_generator, validation_steps=int(len(X_val)/params['batch_size']),
        use_multiprocessing=False,
        callbacks=callbacks_list, initial_epoch = 0)


# In[ ]:




