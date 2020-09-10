import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img

# load an image from file
image = load_img('mug.jpg', target_size=(224, 224))
 
#Load the VGG model
model = VGG16()

vgg_model.summary()
