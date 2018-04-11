from keras.datasets import cifar100
from keras.utils import to_categorical
import numpy as np
from pprint import pprint

from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.preprocessing.image import ImageDataGenerator

seed = 7
np.random.seed(seed)

# Set up dataset specific values
batch_size = 1000
num_classes = 100

# input image dimensions
img_rows, img_cols = 32, 32

# Load dataset
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# (for MLP only) flatten 28*28 images to a 784 vector for each image
#num_pixels = x_train.shape[1] * x_train.shape[2]
#X_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
#X_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

# (for CNNs)
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

# reshape to be [samples][width][height][channels]
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 3).astype('float32')
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 3).astype('float32')

# normalize inputs from 0-255 to 0-1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# define data preparation
#datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
#shift = 0.3
#datagen = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift)
#datagen = ImageDataGenerator(rotation_range=90)
datagen = ImageDataGenerator(contrast_stretching=True, adaptive_equalization=False, histogram_equalization=False)
# fit parameters from data
datagen.fit(x_train)
# configure batch size
for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):
    break
   
    
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
pprint(y_train)
pprint(y_test)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)