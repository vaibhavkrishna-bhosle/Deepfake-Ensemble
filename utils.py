import numpy as np
from tensorflow import keras as keras
import cv2
from PIL import Image
import streamlit as st

def get_image_array(path):
    opencv_image = cv2.imdecode(path, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    return opencv_image


def serial(path):
    model = keras.models.load_model('models/serial.h5')
    image = get_image_array(path)
    image = cv2.resize(image,(224,224))
    X = np.array(image).reshape(-1, 224, 224, 3)
    x = X/255.0
    image = keras.utils.normalize(x,axis=1)
    predictions = model.predict(image)

    return predictions

def vgg16(path):
    model = keras.models.load_model('models/upgraded_vgg16.h5')
    image = get_image_array(path)
    image = cv2.resize(image,(224,224))
    X = np.array(image).reshape(-1, 224, 224, 3)
    x = X/255.0
    image = keras.utils.normalize(x,axis=1)
    predictions = model.predict(image)
    return predictions

def mobilenet(path):
    model = keras.models.load_model('models/mobilenet.h5')
    image = get_image_array(path)
    image = cv2.resize(image,(96,96))
    image = np.array(image).reshape(-1, 96, 96, 3)
    image = image/255.0
    predictions = model.predict(image)
    return predictions

def densenet(path):
    model = keras.models.load_model('models/densenet.h5')
    image = get_image_array(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(224,224))
    X = np.array(image).reshape(-1, 224, 224, 1)
    image = keras.utils.normalize(X,axis=1)
    predictions = model.predict(image)
    return predictions

def clean(out):
    return out[0][0]

def generate_string(out,model):
    out = clean(out)
    out_string = model + ": Their is " + str(round(out*100,2)) + "%" + " chance that the image is real."
    return out_string
    
def cal_ensemble(a,b,c,model):
    out = (clean(a) + clean(b) + clean(c))/4
    out_string = model+ ": Their is " + str(round(out*100,2)) + "%" + " chance that the image is real."
    return  out_string

def display_image(path):
    image = get_image_array(path)
    st.image(image, use_column_width=True)
