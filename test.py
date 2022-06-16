import numpy as np
import h5py
import cv2
import pandas as pd
import imgaug as ia
from imgaug import augmenters as iaa
import gc
from matplotlib import pyplot as plt
import time
from sklearn.metrics import *

from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.initializers import *
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
lbl=[]
img=np.zeros((3064,224,224))
for i in range(1,3065):
    try:
        path='/kaggle/input/brain-tumour/brainTumorDataPublic_1766/'
        with h5py.File(path+str(i)+'.mat') as f:
          images = f['cjdata']
          resized = cv2.resize(images['image'][:,:], (224,224), interpolation = cv2.INTER_CUBIC )
          x=np.asarray(resized)
          x=(x-np.min(x))/(np.max(x)-np.min(x))
          x=x.reshape((1,224,224))
          img[i-1]=x
          lbl.append(int(images['label'][0]))
    except:
        try:
          path='/kaggle/input/brain-tumour/brainTumorDataPublic_22993064/'
          with h5py.File(path+str(i)+'.mat') as f:
              images = f['cjdata']
              resized = cv2.resize(images['image'][:,:], (224,224), interpolation = cv2.INTER_CUBIC )
              x=np.asarray(resized)
              x=(x-np.min(x))/(np.max(x)-np.min(x))
              x=x.reshape((1,224,224))
              img[i-1]=x
              lbl.append(int(images['label'][0]))
        except:
            try:
              path='/kaggle/input/brain-tumour/brainTumorDataPublic_15332298/'
              with h5py.File(path+str(i)+'.mat') as f:
                  images = f['cjdata']
                  resized = cv2.resize(images['image'][:,:], (224,224), interpolation = cv2.INTER_CUBIC )
                  x=np.asarray(resized)
                  x=(x-np.min(x))/(np.max(x)-np.min(x))
                  x=x.reshape((1,224,224))
                  img[i-1]=x
                  lbl.append(int(images['label'][0]))
            except:
              path='/kaggle/input/brain-tumour/brainTumorDataPublic_7671532/'
              with h5py.File(path+str(i)+'.mat') as f:
                  images = f['cjdata']
                  resized = cv2.resize(images['image'][:,:], (224,224), interpolation = cv2.INTER_CUBIC )
                  x=np.asarray(resized)
                  x=(x-np.min(x))/(np.max(x)-np.min(x))
                  x=x.reshape((1,224,224))
                  img[i-1]=x
                  lbl.append(int(images['label'][0]))

path='/kaggle/input/braintumour/cvind (2).mat'

with h5py.File(path) as f:
      data=f['cvind']
      idx=data[0]
import scipy.io
obj_arr = {}
obj_arr['images'] = img
obj_arr['label'] = lbl
obj_arr['fold']=idx
np.save('check.npy', obj_arr)
path = "check.npy"
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]



#change targets
def change(img):
    resized = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA )
    return resized




#get train and test splits
def get_trn_tst(df,tst_fold):
  idx=np.asarray(df['fold'])
  y=np.asarray(df['label'])
  y-=1
  img=np.asarray(df['images'])
  img1=[]
  for i in range(len(img)):
        img1.append(change(img[i]))
  img1=np.asarray(img1)
  del([img])
  gc.collect()
  trn_y=np.asarray(y[(idx!=tst_fold)])
  trn_img=np.asarray(img1[(idx!=tst_fold)])
  tst_y=np.asarray(y[(idx==tst_fold)])
  tst_img=img1[idx==tst_fold]
  trn_img=np.repeat(trn_img.reshape((trn_img.shape[0],224,224,1)),3,axis=3)
  tst_img=np.repeat(tst_img.reshape((tst_img.shape[0],224,224,1)),3,axis=3)
  return (trn_img.copy(),trn_y.copy()),(tst_img.copy(),tst_y.copy())

class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)
    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    def compute_output_shape(self, input_shape):
        return input_shape
class abc(Layer):
    def __init__(self,inr,size,mo,up,org,**kwargs):
        super(abc, self).__init__(**kwargs)
        self.inr=inr
        self.mo=mo
        self.up=up
        self.org=org
        self.size=size
    def get_config(self):
        base_config = super(abc, self).get_config()

    def build(self, input_shape):
        super(abc, self).build(input_shape)
        self.cv1 = Conv2D(self.inr,1)
        self.cv2 = Conv2D(self.inr,1)
        
        
        
        self.dns1 = Conv2D(self.org,1,activation='relu')
        self.dns2 = Conv2D(self.org,1,activation='sigmoid')
        
        
        self.cv3 = Conv2D(1,1)
        self.up = UpSampling2D(interpolation='bilinear',size=(self.up,self.up))
        self.dns1=Dense(1)
    def call(self, img,y):
        y = self.cv1(y)
        x = self.cv2(img)
        y = self.up(y)
        
        y = Add()([y,x])
        y=GlobalAveragePooling2D()(y)
        y = Reshape((1,1,self.inr))(y)
        x = self.dns1(y)
        x = self.dns2(x)
        z = tf.math.multiply(img,x)
        
        x = ReLU()(z)
        x = K.max(x,axis=-1)
        x = Reshape((self.size,self.size,1))(x)
        
        map = softmax(x,axis=[2,3])


        return tf.math.multiply(z,map)


def load_model():   
  
  K.clear_session() 
  mod=DenseNet121(include_top=True, weights='imagenet')
  d = mod.get_layer('conv5_block16_concat').output
  d = Conv2D(512,1)(d)

  a = mod.get_layer('conv3_block12_concat').output
  a = Conv2D(256,1)(a)
  a = abc(inr=16,mo=256,up=4,size=28,org=256)(a,d)

  b = mod.get_layer('conv4_block24_concat').output
  b = Conv2D(512,1)(b)
  b = abc(inr=16,mo=256,up=2,size=14,org=512)(b,d)

  b = LayerNormalization()(b)
  b = Reshape((14,14,512))(b)
  b = UpSampling2D(interpolation='bilinear',size=(2,2))(b)
  d = UpSampling2D(interpolation='bilinear',size=(4,4))(d)
    
  conc=Concatenate(axis=-1)([a,b,d])
  conc = GlobalMaxPooling2D()(conc)

  conc = Dense(3, activation="softmax")(conc) 
  
  mod=Model(inputs=mod.input,outputs=conc)
  return mod




def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result
def Hflip( images):
		seq = iaa.Sequential([iaa.Fliplr(1.0)])
		return seq.augment_images(images)
def Vflip( images):
		seq = iaa.Sequential([iaa.Flipud(1.0)])
		return seq.augment_images(images)
def noise(images):
    ls=[]
    for i in images:
        x = np.random.normal(loc=0, scale=0.05, size=(299,299,3))
        ls.append(i+x)
    return ls
def rotate(images):
    ls=[]
    for angle in range(-15,20,5):
        for image in images:
            ls.append(rotate_image(image,angle))
    return ls
class DataGenerator(tf.keras.utils.Sequence):
  def __init__(self, images, labels, batch_size=64, image_dimensions = (96 ,96 ,3), shuffle=False, augment=False):
    self.labels       = labels              # array of labels
    self.images = images        # array of image paths
    self.batch_size   = batch_size          # batch size
    self.on_epoch_end()

  def __len__(self):
    return int(np.floor(self.labels.shape[0] / self.batch_size))

  def on_epoch_end(self):
    self.indexes = np.arange(self.labels.shape[0])

  def __getitem__(self, index):
		# selects indices of data for next batch
    indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
    # select data and load images
    labels = self.labels.loc[indexes]
    img = [self.images[k].astype(np.float32) for k in indexes]
    imgH=Hflip(img)
    imgV=Vflip(img)
    imgR=rotate(img)
    images=[]
    images.extend(imgH)
    images.extend(imgV)
    images.extend(imgR)
    lbl=labels.copy()
    labels=pd.DataFrame()
    labels=pd.concat([labels,lbl],0)
    labels=pd.concat([labels,lbl],0)
    labels=pd.concat([labels,lbl],0)
    labels=pd.concat([labels,lbl],0)
    labels=pd.concat([labels,lbl],0)
    labels=pd.concat([labels,lbl],0)
    labels=pd.concat([labels,lbl],0)
    labels=pd.concat([labels,lbl],0)
    labels=pd.concat([labels,lbl],0)
    #images = np.array([preprocess_input(img) for img in images])
    return np.asarray(images), np.asarray(labels.values)




def upd(dk,data):
    if dk==0:
        dk=data
    else:
        for ky in data.keys():
            dk[ky].extend(data[ky])
    return dk
def test(index):
  df=np.load(path,allow_pickle=True)
  df=df.item()
  fold='fold_'+str(index)
  trn,tst=get_trn_tst(df,index)
  model=load_model()
  tst_x,tst_y=unison_shuffled_copies(tst[0],tst[1])
  model.load_weights('weights.hdf5')
  pre=model.predict(tst_x)
  pre=np.argmax(pre,1)
  print(accuracy_score(pre,tst_y))
    
index=1
test(index)