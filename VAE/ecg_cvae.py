import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
import sys
import time
import os
from tqdm import tqdm
sys.path.append('../preprocessing/DataSet/')
from DBhandle import DBconn #DBhandle.py 파일 확인
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, optimizers, metrics
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Input, BatchNormalization, Dense, Lambda, Reshape, UpSampling1D,Flatten, Input, LeakyReLU, Activation
from tensorflow.keras.layers import (
    AvgPool1D,
    GlobalAveragePooling1D,
    MaxPool1D,
    Conv1D,
    Conv1DTranspose,
    Layer
)
from tensorflow.keras.layers import ReLU, concatenate

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #초기화할 GPU Number
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.client import device_lib

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#session.close()

db = DBconn("smc_ssd")
df_normal = db.selectTableData('Single_ECG_Normal_PQRST')
df_afib = db.selectTableData('Single_ECG_AFib_PQRST')
df_train = df_normal.append(df_afib)
df_train.reset_index(drop=True,inplace=True)
df_train['group'] = pd.cut(df_train['patient_id'].astype('category').cat.codes,20,labels=list(range(0,20)))
grouped = df_train['group']
df_valid = df_train[((df_train['group'] == 18)|(df_train['group'] == 17)|(df_train['group'] == 16))].copy()
df_train = df_train[((df_train['group'] != 18)&(df_train['group'] != 17)&(df_train['group'] != 16))]
df_train.reset_index(drop=True,inplace=True)
df_valid.reset_index(drop=True,inplace=True)


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, unique_ids, label, batch_size=128, dim=(4980,12),n_classes=2, shuffle=False):
        
        self.dim = dim
        self.batch_size = batch_size
        self.label = label
        self.unique_ids = unique_ids
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = np.arange(len(unique_ids))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.unique_ids) / self.batch_size))

    def __getitem__(self, index):
                           
        if((index+1)*self.batch_size>len(self.unique_ids)): #마지막 batch 데이터
            indexes = self.indexes[index*self.batch_size:len(self.unique_ids)]
        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size] #0 부터 배치 사이즈까지 self.indexes

        list_unique_ids_temp = [self.unique_ids[k] for k in indexes]
        list_label_temp = [self.label[k] for k in indexes]

        # Generate data
        X= self.__data_generation(list_unique_ids_temp,list_label_temp)
        
        return X

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_unique_ids_temp,list_label_temp):
        scaler = StandardScaler()
        # Initialization
        if(len(list_unique_ids_temp)<self.batch_size): #마지막 batch 내 데이터
            last_size = len(self.unique_ids) % self.batch_size
            X = np.empty((last_size, *self.dim))
        else:
            X = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, (ID,label) in enumerate(zip(list_unique_ids_temp, list_label_temp)):
            # Store sample:
            ecg_wave = np.load('/smc_work/datanvme/smc/origin/' + ID + '.npy')
            ecg_wave = scaler.fit_transform(ecg_wave)
            X[i,] = ecg_wave[10:4990,1].reshape(-1,1)
            #X[i,] = ecg_wave[10:4990,:]
            
        return X

params = {'dim': (4980,1),
          'batch_size': 512,
          'n_classes': 2,
          'shuffle': False}

train_dataset = DataGenerator(df_train['unique_id'],df_train['label'], **params)
valid_dataset = DataGenerator(df_valid['unique_id'],df_valid['label'], **params)
#test_generator = DataGenerator(df_test['unique_id'],df_test['label'], **params)

def sampling(args):
    """Reparameterization trick. Instead of sampling from Q(z|X), 
    sample eps = N(0,I) z = z_mean + sqrt(var)*eps.

    Parameters:
    -----------
    args: list of Tensors
        Mean and log of variance of Q(z|X)

    Returns
    -------
    z: Tensor
        Sampled latent vector
    """

    z_mean, z_log_var = args
    eps = tf.random.normal(tf.shape(z_log_var), dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')
    z = z_mean + tf.exp(z_log_var / 2) * eps
    return z
  
  
z_dim = 256

# The Encoder 
# ************
encoder_input = Input(shape=(4980,1))
x = encoder_input
x = Conv1D(filters = 32, kernel_size = 4, strides = 1, padding='same')(x)
x = LeakyReLU()(x)
x = Conv1D(filters = 32, kernel_size = 4, strides = 1, padding='same')(x)
x = LeakyReLU()(x)
x = Conv1D(filters = 64, kernel_size = 4, strides = 2, padding='same')(x)
x = LeakyReLU()(x)
x = Conv1D(filters = 64, kernel_size = 4, strides = 2, padding='same')(x)
x = LeakyReLU()(x)
x = Conv1D(filters = 128, kernel_size = 4, strides = 2, padding='same')(x)
x = LeakyReLU()(x)
x = Conv1D(filters = 128, kernel_size = 4, strides = 2, padding='same')(x)
x = LeakyReLU()(x)
x = MaxPool1D(4)(x)

shape_before_flattening = K.int_shape(x)[1:]  # B is the tensorflow.keras backend ! See last post. 

x = Flatten()(x)

# differences to AE-models. The following layers central elements of VAEs!   
mu      = Dense(z_dim, name='mu')(x)
log_var = Dense(z_dim, name='log_var')(x)

# We calculate z-points/vectors in the latent space by a special function
# used by a Keras Lambda layer   
enc_out = Lambda(sampling, name='enc_out_z')([mu, log_var])    

# The Encoder model 
encoder = Model(encoder_input, [enc_out], name="encoder")
encoder.summary()

dec_inp_z = Input(shape=(z_dim))
x = Dense(np.prod(shape_before_flattening))(dec_inp_z)
x = Reshape(shape_before_flattening)(x)
x = Conv1DTranspose(filters=64, kernel_size=4, strides=2, padding='same')(x)
x = LeakyReLU()(x) 
x = Conv1DTranspose(filters=64, kernel_size=4, strides=2, padding='same')(x)
x = LeakyReLU()(x) 
x = Conv1DTranspose(filters=32, kernel_size=5, strides=2, padding='same')(x)
x = LeakyReLU()(x) 
x = Conv1DTranspose(filters=32, kernel_size=5, strides=2, padding='same')(x)
x = LeakyReLU()(x) 
x = Conv1DTranspose(filters=1,  kernel_size=5, strides=1, padding='same')(x)
x = LeakyReLU()(x) 
x = Conv1DTranspose(filters=1,  kernel_size=5, strides=1, padding='same')(x)
# Output - 
x = Activation('sigmoid')(x)
dec_out = x
decoder = Model([dec_inp_z], [dec_out], name="decoder")

decoder.summary()

enc_output = encoder(encoder_input)
decoder_output = decoder(enc_output)
vae_pre = Model(encoder_input, decoder_output, name="vae_witout_kl_loss")

class CustVariationalLayer (Layer):
    
    def vae_loss(self, x_inp_ecg, z_reco_ecg):
        # The references to the layers are resolved outside the function 
        x = K.flatten(x_inp_ecg)   # B: tensorflow.keras.backend
        z = K.flatten(z_reco_ecg)
        
        # reconstruction loss per sample 
        # Note: that this is averaged over all features (e.g.. 784 for MNIST) 
        reco_loss = metrics.binary_crossentropy(x, z)
        # KL loss per sample - we reduce it by a factor of 1.e-3 
        # to make it comparable to the reco_loss  
        kln_loss  = -0.5 * K.mean(1 + log_var - K.square(mu) - K.exp(log_var), axis=1) 
        # mean per batch (axis = 0 is automatically assumed) 
        return K.mean(reco_loss + kln_loss), K.mean(reco_loss), K.mean(kln_loss) 
           
    def call(self, inputs):
        inp_ecg = inputs[0]
        out_ecg = inputs[1]
        total_loss, reco_loss, kln_loss = self.vae_loss(inp_ecg, out_ecg)
        self.add_loss(total_loss, inputs=inputs)
        self.add_metric(total_loss, name='total_loss', aggregation='mean')
        self.add_metric(reco_loss, name='reco_loss', aggregation='mean')
        self.add_metric(kln_loss, name='kl_loss', aggregation='mean')
        
        return out_ecg  #not really used in this approach  
      
      
enc_output = encoder(encoder_input)
decoder_output = decoder(enc_output)

# add the custom layer to the model  
fc = CustVariationalLayer()([encoder_input, decoder_output])

vae = Model(encoder_input, fc, name="vae")
vae.summary()

vae.compile(optimizer=Adam(), loss=None)

X_train, X_test = np.array(train_dataset[1],dtype='float32'), np.array(valid_dataset[1],dtype='float32')

n_epochs = 50
batch_size = 128
vae.fit( x=X_train, y=None, shuffle=True, 
         epochs = n_epochs, batch_size=batch_size,validation_data=(X_test,None))

