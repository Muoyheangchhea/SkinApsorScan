import os
from typing import List
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from models.skin_tone.skin_tone_knn import identify_skin_tone
from flask import Flask, request
from flask_restful import Api, Resource, reqparse, abort
import werkzeug
from models.recommender.rec import recs_essentials, makeup_recommendation
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
api = Api(app)

class_names1 = ['Dry_skin', 'Normal_skin', 'Oil_skin']
class_names2 = ['Low', 'Moderate', 'Severe']
skin_tone_dataset = 'models/skin_tone/skin_tone_dataset.csv'

def get_model():
    global model1, model2
    try:
        model1 = load_model('./models/skin_model.h5')  
        print('Model 1 loaded')
    except Exception as e:
        print(f"Error loading Model 1: {e}")

    try:
        model2 = load_model('./models/acne_model.h5')  
        print("Model 2 loaded!")
    except Exception as e:
        print(f"Error loading Model 2: {e}")

def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.0  # Normalize pixel values
    return img_tensor

def prediction_skin(img_path):
    new_image = load_image(img_path)
    pred1 = model1.predict(new_image)
    if len(pred1[0]) > 1:
        pred_class1 = class_names1[tf.argmax(pred1[0])]
    else:
        pred_class1 = class_names1[int(tf.round(pred1[0]))]
    return pred_class1

def prediction_acne(img_path):
    new_image = load_image(img_path)
    pred2 = model2.predict(new_image)
    if len(pred2[0]) > 1:
        pred_class2 = class_names2[tf.argmax(pred2[0])]
    else:
        pred_class2 = class_names2[int(tf.round(pred2[0]))]
    return pred_class2

get_model()

img_put_args = reqparse.RequestParser()
img_put_args.add_argument(
    "file", help="Please provide a valid image file", required=True)

rec_args = reqparse.RequestParser()
rec_args.add_argument(
    "tone", type=int, help="Argument required", required=True)
rec_args.add_argument(
    "type", type=str, help="Argument required", required=True)
rec_args.add_argument("features", type=dict,
                      help="Argument required", required=True)

class SkinMetrics(Resource):
    def put(self):
        # Parse image file input
        args = img_put_args.parse_args()
        file = args['file']
        starter = file.find(',')
        image_data = file[starter+1:]
        image_data = bytes(image_data, encoding="ascii")
        im = Image.open(BytesIO(base64.b64decode(image_data + b'==')))

        # Save the image
        filename = 'image.png'
        file_path = os.path.join('./static', filename)
        im.save(file_path)

        # Use AI models to predict skin type, tone, and acne severity
        skin_type = prediction_skin(file_path).split('_')[0]
        acne_type = prediction_acne(file_path)
        tone = identify_skin_tone(file_path, dataset=skin_tone_dataset)

        # Return the detected metrics
        return {'type': skin_type, 'tone': str(tone), 'acne': acne_type}, 200

class Recommendation(Resource):
    def put(self):
        # Parse image file input
        args = img_put_args.parse_args()
        file = args['file']
        starter = file.find(',')
        image_data = file[starter+1:]
        image_data = bytes(image_data, encoding="ascii")
        im = Image.open(BytesIO(base64.b64decode(image_data + b'==')))

        # Save the image
        filename = 'image.png'
        file_path = os.path.join('./static', filename)
        im.save(file_path)

        # Use AI models to predict skin type, tone, and acne severity
        skin_type = prediction_skin(file_path).split('_')[0]
        acne_type = prediction_acne(file_path)
        tone = identify_skin_tone(file_path, dataset=skin_tone_dataset)

        # Convert AI results into format expected by recommendation system
        features = {
            "normal": 1 if skin_type == "Normal" else 0,
            "dry": 1 if skin_type == "Dry" else 0,
            "oily": 1 if skin_type == "Oil" else 0,
            "combination": 1,  # Default to combination
            "acne": 1 if acne_type in ["Moderate", "Severe"] else 0,
            "sensitive": 0,  # You can determine more features based on the logic you want
            "fine lines": 0,
            "wrinkles": 0,
            "redness": 0,
            "dull": 0,
            "pore": 0,
            "pigmentation": 0,
            "blackheads": 0,
            "whiteheads": 0,
            "blemishes": 0,
            "dark circles": 0,
            "eye bags": 0,
            "dark spots": 0
        }

        # Use the tone to determine skin tone description
        if tone <= 2:
            skin_tone = 'fair to light'
        elif tone >= 4:
            skin_tone = 'medium to dark'
        else:
            skin_tone = 'light to medium'

        # Run recommendation system
        fv = [int(value) for key, value in features.items()]
        general = recs_essentials(fv, None)
        makeup = makeup_recommendation(skin_tone, skin_type.lower())

        return {'general': general, 'makeup': makeup}

api.add_resource(SkinMetrics, "/upload")
api.add_resource(Recommendation, "/recommend")

if __name__ == "__main__":
    app.run(debug=False)
