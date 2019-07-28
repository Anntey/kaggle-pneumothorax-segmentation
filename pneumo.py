
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config = config)

#############
# Libraries #
#############

#!pip install -U efficientnet==0.0.4
#from efficientnet import EfficientNetB4
import sys
import cv2
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
import glob
import shutil
import os
import random
from PIL import Image
from tqdm import tqdm
from mask_functions import mask2rle
from sklearn.model_selection import train_test_split
from keras import Model
from keras import backend as K
from keras.utils import plot_model
from keras.applications.xception import Xception
from keras.losses import binary_crossentropy
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import (Input, Conv2D, Conv2DTranspose, concatenate, Dropout, 
                          BatchNormalization, LeakyReLU, ZeroPadding2D, Add)
from albumentations import (Compose, HorizontalFlip, RandomBrightness, RandomContrast, RandomGamma, OneOf,
                            GridDistortion, ElasticTransform, OpticalDistortion, RandomSizedCrop, CLAHE)

###############
# Random seed #
###############

seed = 10
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

################
# Loading data #
################

!mkdir ./input/masks
!mkdir ./input/test
!mkdir ./input/train
!mkdir ./input/img_train
!mkdir ./input/mask_train
!mkdir ./input/img_val
!mkdir ./input/mask_val

!unzip -q ./input/masks.zip -d ./input/masks 
!unzip -q ./input/train.zip -d ./input/train 
!unzip -q ./input/test.zip -d ./input/test

img_size = 256
batch_size = 16

mask_paths_all = glob.glob("./input/masks/*")
mask_df = pd.DataFrame()
mask_df["file_name"] = mask_paths_all
mask_df["label"] = 0

for path in mask_paths_all:
    if np.array(Image.open(path)).sum() > 0:
        mask_df.loc[mask_df["file_name"] == path, "label"] = 1 # label positive if pneumothorax present

train_img_paths_all = glob.glob("./input/train/*")
test_img_paths_all = glob.glob("./input/test/*")

train_img_paths, val_img_paths = train_test_split(
    train_img_paths_all,
    stratify = mask_df["label"],
    test_size = 0.1,
    random_state = seed
)

train_mask_paths = [path.replace("train", "masks") for path in train_img_paths] # get corresponding mask paths
val_mask_paths = [path.replace("train", "masks") for path in val_img_paths]

for path in train_img_paths:
    filename = path.split("/")[-1] # split path string and get filename from last index
    shutil.move(path, "./input/img_train/" + filename)
    
for path in train_mask_paths:
    filename = path.split("/")[-1]
    shutil.move(path, "./input/mask_train/" + filename)
    
for path in val_img_paths:
    filename = path.split("/")[-1]
    shutil.move(path, "./input/img_val/" + filename)
    
for path in val_mask_paths:
    filename = path.split("/")[-1]
    shutil.move(path, "./input/mask_val/" + filename) 

x_test = list(cv2.resize(np.array(Image.open(path)), (img_size, img_size)) for path in test_img_paths_all) # open and resize test images
x_test = np.array(x_test) # convert list to (n, h, w) array
x_test = np.array([np.repeat(img[..., None], repeats = 3, axis = 2) for img in x_test]) # reshape to (n, h, w, ch) array
x_test = x_test / 255 # scale to 0...1

#################
# Augmentations #
#################

augs_train = Compose([
    HorizontalFlip(p = 0.5),
    OneOf([
        RandomContrast(),
        RandomGamma(),
        RandomBrightness(),
         ], p = 0.3),
    OneOf([
        ElasticTransform(alpha = 120, sigma = 120 * 0.05, alpha_affine = 120 * 0.03),
        GridDistortion(),
        OpticalDistortion(distort_limit = 2, shift_limit = 0.5),
        ], p = 0.3),
    RandomSizedCrop(min_max_height = (160, 256), height = img_size, width = img_size, p = 0.25),
    CLAHE(p = 1),
], p = 1)


augs_test = Compose([
    CLAHE(p = 1),
], p = 1)
    
######################
# Generator function #
######################

class DataGenerator(keras.utils.Sequence):

    def __init__(self, img_path = None, mask_path = None, augmentations = None,
                 batch_size = 16, img_size = 256, n_channels = 3, shuffle = False):
        self.batch_size = batch_size
        self.mask_path = mask_path
        self.img_path = img_path
        self.img_paths_all = glob.glob(img_path + "/*")      
        self.img_size = img_size 
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augmentations
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.img_paths_all) / self.batch_size)) # number of batches per epoch

    def __getitem__(self, index): # generates one batch
        indices = self.indices[index * self.batch_size:min((index + 1) * self.batch_size, len(self.img_paths_all))] # get indices for batch
        img_ids_list = [self.img_paths_all[k] for k in indices] # get list of image IDs
        X, y = self.data_generation(img_ids_list) # generate data

        if self.augment is None:
            return (np.array(X) / 255), (np.array(y) / 255) # scale to 0...1 range
        else:            
            img, mask = [], []   
            for x, y in zip(X, y):
                augmented = self.augment(image = x, mask = y)
                img.append(augmented["image"])
                mask.append(augmented["mask"])
            return (np.array(img) / 255), (np.array(mask) / 255) # augmentations and scale

    def on_epoch_end(self):
        self.indices = np.arange(len(self.img_paths_all)) # updates indices after each epoch
        
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def data_generation(self, img_ids_list):  # generate batches (batch_size, h, w, ch)
        X = np.empty((len(img_ids_list), self.img_size, self.img_size, self.n_channels))
        y = np.empty((len(img_ids_list), self.img_size, self.img_size, 1))

        for i, img_path in enumerate(img_ids_list):                 
            mask_path = img_path.replace(self.img_path, self.mask_path)       
            mask = np.array(Image.open(mask_path)) # open mask
            img = np.array(Image.open(img_path)) # open image
            
            if len(img.shape) == 2:
                img = np.repeat(img[..., None], repeats = 3, axis = 2) # reshape image array from (h, w) to (h, w, ch)
                
            X[i, ] = cv2.resize(img, (self.img_size, self.img_size)) # resize image and store in (batch_size, h, w, ch) array
            y[i, ] = cv2.resize(mask, (self.img_size, self.img_size))[..., np.newaxis] # resize mask and store (batch_size, h, w, 1) array
            y[y > 0] = 255 # store class

        return np.uint8(X), np.uint8(y) # augmentations library requires uint8

###################################
# Train and validation generators #
###################################

train_gen = DataGenerator(
    img_path = "./input/img_train",
    mask_path = "./input/mask_train",
    augmentations = augs_train,
    img_size = img_size,
    batch_size = batch_size,
    shuffle = True
)

val_gen = DataGenerator(
    img_path = "./input/img_val",
    mask_path = "./input/mask_val",
    augmentations = augs_test,
    img_size = img_size,
    batch_size = batch_size,
    shuffle = True
)

#foo, bar = train_gen.__getitem__(2)
#print(foo.shape, bar.shape)
#foo, bar = val_gen.__getitem__(2)
#print(foo.shape, bar.shape)

######################
# Visualizing images #
######################

eg_gen = DataGenerator( # generator without augmentations
        img_path = "./input/img_train",
        mask_path = "./input/mask_train",
        batch_size = 8,
        augmentations = augs_test,
        shuffle = False
)

eg_gen_aug = DataGenerator( # with augmentations
        img_path = "./input/img_train",
        mask_path = "./input/mask_train",
        batch_size = 8,
        augmentations = augs_train,
        shuffle = False
) 

img_eg, mask_eg = eg_gen.__getitem__(0) # get one pair of img + mask from generator

fig, axs = plt.subplots(nrows = 4, ncols = 2, figsize = (10, 20))
for i, (img, mask) in enumerate(zip(img_eg, mask_eg)): # visualize 8 images without augmentations
    ax = axs[int(i / 2), i % 2]
    ax.imshow(img.squeeze())
    ax.imshow(mask.squeeze(), alpha = 0.15, cmap = "Reds")   
    
img_eg_aug, mask_eg_aug = eg_gen_aug.__getitem__(0)

fig, axs = plt.subplots(nrows = 4, ncols = 2, figsize = (10, 20)) 
for i, (img, mask) in enumerate(zip(img_eg_aug, mask_eg_aug)): # with augmentations
    ax = axs[int(i / 2), i % 2]
    ax.imshow(img.squeeze())
    ax.imshow(mask.squeeze(), alpha = 0.15, cmap = "Reds") 

#####################
# Evaluation metric #
#####################

def get_iou_vector(A, B):  
    batch_size = A.shape[0]
    metric = 0.0

    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        true = np.sum(t)
        pred = np.sum(p)

        if true == 0:
            metric += (pred == 0)
            continue        

        intersection = np.sum(t * p)
        union = true + pred - intersection
        iou = intersection / union     
        iou = np.floor(max(0, (iou - 0.45) * 20)) / 10 
        metric += iou
        
    metric /= batch_size
    return metric


def iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred > 0.5], tf.float64)

#################
# Loss function #
#################

#def dice_coef(y_true, y_pred):
#    y_true_f = K.flatten(y_true)
#    y_pred = K.cast(y_pred, "float32")
#    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), "float32")
#    intersection = y_true_f * y_pred_f
#    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
#    return score

def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2.0 * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1.0 - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

####################
# Specifying model #
####################
   
def convolution_block(x, filters, size, strides = (1, 1), padding = "same", activation = True):
    x = Conv2D(filters, size, strides = strides, padding = padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha = 0.1)(x)
    return x

def residual_block(blockInput, num_filters = 16):
    x = LeakyReLU(alpha = 0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3, 3) )
    x = convolution_block(x, num_filters, (3, 3), activation = False)
    x = Add()([x, blockInput])
    return x

inp = Input(shape = (img_size, img_size, 3))

model_base = Xception(
        input_tensor = inp,
        weights = "./input/xception_weights_tf_dim_ordering_tf_kernels_notop.h5",
        include_top = False
)

start_neurons = 16

conv4 = model_base.layers[121].output
conv4 = LeakyReLU(alpha = 0.1)(conv4)
pool4 = MaxPooling2D((2, 2))(conv4)
pool4 = Dropout(0.1)(pool4)

convm = Conv2D(start_neurons * 32, (3, 3), activation = None, padding = "same")(pool4)
convm = residual_block(convm,start_neurons * 32)
convm = residual_block(convm,start_neurons * 32)
convm = LeakyReLU(alpha = 0.1)(convm)

deconv4 = Conv2DTranspose(start_neurons * 16, (3, 3), strides = (2, 2), padding = "same")(convm)
uconv4 = concatenate([deconv4, conv4])
uconv4 = Dropout(0.1)(uconv4)

uconv4 = Conv2D(start_neurons * 16, (3, 3), activation = None, padding = "same")(uconv4)
uconv4 = residual_block(uconv4,start_neurons * 16)
uconv4 = residual_block(uconv4,start_neurons * 16)
uconv4 = LeakyReLU(alpha = 0.1)(uconv4)

deconv3 = Conv2DTranspose(start_neurons * 8, (3, 3), strides = (2, 2), padding = "same")(uconv4)
conv3 = model_base.layers[31].output
uconv3 = concatenate([deconv3, conv3])    
uconv3 = Dropout(0.1)(uconv3)

uconv3 = Conv2D(start_neurons * 8, (3, 3), activation = None, padding = "same")(uconv3)
uconv3 = residual_block(uconv3,start_neurons * 8)
uconv3 = residual_block(uconv3,start_neurons * 8)
uconv3 = LeakyReLU(alpha = 0.1)(uconv3)

deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides = (2, 2), padding = "same")(uconv3)
conv2 = model_base.layers[21].output
conv2 = ZeroPadding2D(((1,0),(1,0)))(conv2)
uconv2 = concatenate([deconv2, conv2])
    
uconv2 = Dropout(0.1)(uconv2)
uconv2 = Conv2D(start_neurons * 4, (3, 3), activation = None, padding = "same")(uconv2)
uconv2 = residual_block(uconv2,start_neurons * 4)
uconv2 = residual_block(uconv2,start_neurons * 4)
uconv2 = LeakyReLU(alpha = 0.1)(uconv2)

deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides = (2, 2), padding = "same")(uconv2)
conv1 = model_base.layers[11].output
conv1 = ZeroPadding2D(((3,0),(3,0)))(conv1)
uconv1 = concatenate([deconv1, conv1])

uconv1 = Dropout(0.1)(uconv1)
uconv1 = Conv2D(start_neurons * 2, (3, 3), activation = None, padding = "same")(uconv1)
uconv1 = residual_block(uconv1,start_neurons * 2)
uconv1 = residual_block(uconv1,start_neurons * 2)
uconv1 = LeakyReLU(alpha = 0.1)(uconv1)

uconv0 = Conv2DTranspose(start_neurons * 1, (3, 3), strides = (2, 2), padding = "same")(uconv1)   
uconv0 = Dropout(0.1)(uconv0)
uconv0 = Conv2D(start_neurons * 1, (3, 3), activation = None, padding = "same")(uconv0)
uconv0 = residual_block(uconv0,start_neurons * 1)
uconv0 = residual_block(uconv0,start_neurons * 1)
uconv0 = LeakyReLU(alpha = 0.1)(uconv0)

uconv0 = Dropout(0.1/2)(uconv0)
output = Conv2D(1, (1,1), padding = "same", activation = "sigmoid")(uconv0)
    
model = Model(inp, output)

model.compile(
    loss = bce_dice_loss,
    optimizer = Adam(lr = 1e-4),
    metrics = [iou_metric]
)
       
plot_model(model, to_file = "model_graph.png") # visualize model architecture

#################
# Fitting model #
#################

callbacks_list = [
#        EarlyStopping(
#                monitor = "val_iou_metric",
#                patience = 10,
#                mode = "max",
#                restore_best_weights = False,
#                verbose = 1
#        ),
        ReduceLROnPlateau(
                monitor = "val_loss",
                factor = 0.5,
                patience = 8,
                mode = "max",
        ),
        ModelCheckpoint(
                filepath = "model_best.h5",
                monitor = "val_iou_metric",
                save_best_only = True,
                save_weights_only = True,
                mode = "max",
                verbose = 1
        )                
]

fit_log = model.fit_generator(
    generator = train_gen,
    validation_data = val_gen,
    shuffle = True,
    epochs = 10,
    #verbose = 2,
    #callbacks = callbacks_list
)

##############
# Evaluation #
##############

fit_log_df = pd.DataFrame(fit_log.history)
fit_log_df[["iou_metric", "val_iou_metric"]].plot()
fit_log_df[["loss", "val_loss"]].plot()

#############################
# Validation set prediction #
#############################

model.load_weights("model_best.h5")

val_gen_pred = DataGenerator(
    img_path = "./input/img_val",
    mask_path = "./input/mask_val",
    augmentations = augs_test,
    img_size = img_size,
    batch_size = 8,
    shuffle = False
)

preds_val = model.predict_generator(val_gen_pred) # get predicted masks

img_pred, mask_pred = val_gen_pred.__getitem__(0) # get (image, actual mask) pairs

fig, axs = plt.subplots(nrows = 4, ncols = 2, figsize = (10, 20))
for i, (img, mask) in enumerate(zip(img_eg, mask_eg)):
    mask_pred = np.round(preds_val[i] > 0.5, decimals = 0) # round to 1 if prediction with > 0.5 confidence
    mask_pred = np.array(mask_pred, dtype = np.float64) # convert to supported data type
    ax = axs[int(i / 2), i % 2]
    ax.imshow(img.squeeze())
    ax.imshow(mask.squeeze(), alpha = 0.15, cmap = "Reds") 
    ax.imshow(mask_pred.squeeze(), alpha = 0.3, cmap = "Greens")

#######################
# Test set prediction #
#######################

threshold = 0.9 # set threshold (obtained by optimizing evaluation metric vs different threshold values)

preds_test = model.predict(x_test, batch_size = batch_size)
    
sys.path.insert(0, "../input/")

masks_rle = []
for pred in tqdm(preds_test): 
    pred = pred.squeeze() # convert (256, 256, 1) array to (256, 256)
    img = cv2.resize(pred, (1024, 1024)) # resize to the original size
    img = img > threshold
    if img.sum() < 1024 * 2: # zero out the smaller regions
        img[:] = 0
    img = (img * 255).astype(np.uint8) # re-scale to 1...255
    masks_rle.append(mask2rle(img, 1024, 1024)) # image compression (RLE)
        
img_ids = list(path.split("/")[-1][:-4] for path in test_img_paths_all) # get filename and remove .png    
    
subm_df = pd.DataFrame({"ImageId": img_ids, "EncodedPixels": masks_rle})
subm_df.loc[subm_df["EncodedPixels"] == "", "EncodedPixels"] = "-1" # set -1 for negative prediction
subm_df.to_csv("submission.csv", index = False)
