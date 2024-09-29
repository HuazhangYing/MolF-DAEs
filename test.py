from molmap.model import RegressionEstimator, MultiClassEstimator, MultiLabelEstimator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from molmap import dataset
from sklearn.utils import shuffle 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from molmap import MolMap
from molmap import feature
# 打开molmap的环境，molmap.model;molmap等宏包会自动导入到这个环境中
# from tensorflow.keras.datasets import mnist
import tensorflow as tf
import os
from tensorflow.keras.layers import Input,Dense, Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, UpSampling2D, Reshape
from tensorflow.keras.models import Model,Sequential
from joblib import load,dump
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from joblib import load,dump

from keras.backend import set_session
from keras.backend import clear_session
from keras.backend import get_session
import tensorflow as tf
import gc
 
gpuid = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
physical_gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_gpus[0], True)

model_path = "../model/maccsfp_test_9_model_best" # replaced by your fingerprint channel.
test_path = '../dataset/test.csv'
# Reset Keras Session
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()
 
    try:
        del classifier # this is from global space
    except:
        pass
 
    print(gc.collect()) # if it does something you should see a number as output
 
    # use the same config as you used to create the session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tf.compat.v1.Session(config=config))

class Encoder(Model):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.d1 = Dense(1024, activation='relu')
        self.d2 = Dense(512, activation='relu')
        self.d3 = Dense(128, activation='relu')
        self.d4 = Dense(64, activation='relu')
        self.d5 = Dense(32, activation='relu')
        self.d6 = Dense(3, activation='relu')
        
    def call(self,x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.d5(x)
        return self.d6(x)       
    
    
class Decoder(Model):   #  pubchem
    def __init__(self):
        super().__init__()
        self.d7 = Dense(32, activation='relu')
        self.d8 = Dense(64, activation='relu')
        self.d9 = Dense(128, activation='relu')
        self.d10 = Dense(512, activation='relu')
        self.d11 = Dense(1024, activation='relu')
        self.d12 = Dense(729, activation='sigmoid')
        self.re = Reshape((27,27))
    
    def call(self,x):
        x = self.d7(x)
        x = self.d8(x)
        x = self.d9(x)
        x = self.d10(x)
        x = self.d11(x)
        x = self.d12(x)
        return  self.re(x)
        
class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def call(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    
model = tf.saved_model.load(model_path)
custom_objects = {
    'Autoencoder': Autoencoder,
    'Encoder': Encoder,
    'Decoder': Decoder
}

import molmap

metric = 'cosine'
method = 'umap'
n_neighbors = 30
min_dist = 0.1

mp_name = 'fingerprint.mp'
bitsinfo = molmap.feature.fingerprint.Extraction().bitsinfo
flist = bitsinfo[bitsinfo.Subtypes.isin(['PubChemFP'])].IDs.tolist()
mp2 = molmap.MolMap(ftype = 'fingerprint', metric = metric, flist = flist)
mp2.fit(method = method, n_neighbors = n_neighbors, min_dist = min_dist)


data1 = pd.read_csv(test_path)
data2 = data1.iloc[:, 8].tolist()

print(data2)
X2 = mp2.batch_transform(data2)


# with open('/mnt/yhz/project/unsupervised/data/ulk1-known/train5-1_Pubchem.data2', 'rb') as file:
#     X2 = pickle.load(file)

loaded_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
print(loaded_model.summary())

#   Real image
fig = plt.figure(figsize = (20,8))
for i in range(10):
    ax = plt.subplot(2,5,i+1)
    ax.imshow(X2[i])
plt.savefig('/mnt/yhz/project/unsupervised/data/ulk1-known/raw_images.png')

#   Predicted image
y_pre = loaded_model.predict(X2)
fig = plt.figure(figsize = (20,8))
for i in range(10):
    ax = plt.subplot(2,5,i+1)
    ax.imshow(y_pre[i])
plt.savefig('/mnt/yhz/project/unsupervised/data/ulk1-known/predicted_images.png')

#   Coordinate mapping
X3 = loaded_model.encoder(X2)
print(X3)
X3_df = pd.DataFrame(X3)
df = pd.DataFrame(X3.numpy(), columns=['X', 'Y', 'Z'])
df.to_csv('/mnt/yhz/project/unsupervised/data/ulk1-known/ulk1-demo_Pubchem.txt', sep='\t', index=False, header=False)