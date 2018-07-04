
import keras
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
# load an image from file
image = load_img('mug.jpg', target_size=(224, 224))
 
#Load the VGG model
model = VGG16()

vgg_model.summary()
