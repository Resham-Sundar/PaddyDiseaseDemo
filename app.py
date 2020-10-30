# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 17:55:53 2020

@author: Resham Sundar
"""
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

MODEL_PATH = 'model.weights.best3.hdf5'
model = load_model(MODEL_PATH)
cwd = os.getcwd()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded'

@app.route('/') 
def upload_f(): 
    return render_template('upload.html') 

def finds(): 
    test_datagen = ImageDataGenerator(rescale = 1./255) 
    vals = ['Brown Leaf Spot', 'Brown Plant Hopper' ,'False Smut'] # change this according to what you've trained your model to do 
    test_dir = cwd
    test_generator = test_datagen.flow_from_directory( 
            test_dir, 
            target_size =(224, 224), 
            color_mode ="rgb", 
            shuffle = False, 
            class_mode ='categorical', 
            batch_size = 1) 
  
    pred = model.predict_generator(test_generator) 
    print(pred) 
    return str(vals[np.argmax(pred)])

def removeAll():
    files = glob.glob('uploaded/*')
    for f in files:
        os.remove(f)

@app.route('/uploader', methods = ['GET', 'POST']) 
def upload_file(): 
    if request.method == 'POST': 
        f = request.files['file'] 
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))) 
        val = finds() 
        removeAll()
        return render_template('pred.html', ss = val)
    
if __name__ == '__main__': 
    app.run()
