import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import io
import pathlib

IMAGE_SIZE = 192
# Load your trained model1
cnn_model = tf.keras.models.load_model('../cnn.keras')
mobile_net_model = tf.keras.models.load_model('../mobile_net_debit.keras')
class_names = ['debit-card', 'something']


def make_predictions(img_array):
    col1, col2 = st.columns(2)

    predictions = mobile_net_model.predict(img_array)
    score = tf.nn.sigmoid(predictions)
    predictions = tf.where(score < 0.5, 0, 1)[0]
    c = class_names[predictions[0]]
    with col1:
        st.header("MobileNetV2")
        st.write("Class: {} - {}, cut-off: {:.2f}".format(
            predictions[0], c, score[0][0]
        ))

    predictions = cnn_model.predict(img_array)
    score = tf.nn.sigmoid(predictions)
    predictions = tf.where(score < 0.5, 0, 1)[0]
    c = class_names[predictions[0]]
    with col2:
        st.header("CNN Network")
        st.write("Class: {} - {}, cut-off: {:.2f}".format(
            predictions[0], c, score[0][0]
        ))


uploaded_file = st.file_uploader("Choose a image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(bytes_data))
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = img_to_array(image)
    img_array = img.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
    make_predictions(img_array)
