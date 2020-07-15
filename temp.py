from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = r'C:/Users/nikit/Documents/nikita/mycnn.h5'

# Load your trained model
#model = tf.keras.models.load_model(MODEL_PATH,compile=False)
#model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.mycnn import mycnn
#model = mycnn(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

class_names=["Burnt_Pizza","Good_Pizza"]

#def prepare(filepath):
   #IMG_SIZE=32
   #img_array=cv2.imread(filepath)
   #new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
   #return new_array.reshape((-1,IMG_SIZE,IMG_SIZE,3))


def model_predict(img_path,MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH,compile=False)
    img = image.load_img(img_path, target_size=(32, 32))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')
    x=x.reshape(-1,32,32,3)

    preds = model.predict_classes(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    
        print("starting prediction")
        if request.method == 'POST':
        # Get the file from post request
            f = request.files['file']
            # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        preds = model_predict(file_path,MODEL_PATH)
        # Process your result for human
        #pred_class = preds.argmax(axis=-1)  
        #prediction=model.predict_classes([prepare('cheese-burst-pizza.jpg')])
        #print(class_names[int(prediction)])
          # Simple argmax
        #pred_class = decode_predictions(preds,top=1)
        print('my pred=',class_names[int(preds)])
        return class_names[int(preds)]
        


if __name__ == '__main__':
    app.run(debug=True)
