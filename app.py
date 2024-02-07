from flask import Flask, request, jsonify, redirect, url_for
from flasgger import Swagger, swag_from
import base64
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import io
import pathlib
import os
import json

IMAGE_SIZE = 192
# Load your trained model
mobile_net_model = tf.keras.models.load_model('model/mobile_net_debit.keras')

app = Flask(__name__)
swagger = Swagger(app, template={
    "swagger": "2.0",
    "info": {
        "title": "My API",
        "description": "API Documentation",
        "version": "1.0.0"
    },
    "basePath": "/",
})

API_KEY = 'check24at'  # Replace with your actual API key


@app.route('/')
def index():
    return redirect(url_for('flasgger.apidocs'))


@app.route('/debit-card-check', methods=['POST'])
@swag_from({
    'parameters': [
        {
            'name': 'payload',
            'in': 'body',
            'required': 'true',
            'schema': {
                'type': 'object',
                'properties': {
                    'fileType': {
                        'type': 'string',
                        'enum': ['JPG', 'PNG'],
                        'example': 'PNG',
                        'description': 'File type of the image'
                    },
                    'image': {
                        'type': 'string',
                        'example': 'base64 encoded image string',
                        'description': 'Base64 encoded image'
                    }
                }
            }
        }
    ],
    'responses': {
        '200': {
            'description': 'Upload successful',
            'schema': {
                'type': 'object',
                'properties': {
                    'accepted': {
                        'type': 'bool',
                        'example': True
                    }
                }
            }
        },
        '400': {
            'description': 'No image provided or an error occurred',
        },
        '401': {
            'description': 'Unauthorized',
        },
    },
})
def check_for_debit_card():
    # Check if an image file was posted

    image_b64 = request.json['image']
    file_type = request.json['fileType']
    try:
        # Decode the base64 image
        bytes_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(bytes_data))
        image = image.convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        img = img_to_array(image)
        img_array = img.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
        predictions = mobile_net_model.predict(img_array)
        score = tf.nn.sigmoid(predictions)
        predictions = tf.where(score < 0.5, 0, 1)[0]
        p = predictions.numpy()[0]
        return jsonify({"accepted": bool(p == 0)}), 200

    except Exception as e:
        return jsonify({'message': 'An error occurred: ' + str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=8080)
