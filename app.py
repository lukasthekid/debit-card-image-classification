from flask import Flask, request, jsonify, redirect, url_for
from flask_restx import Resource, Api, fields
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.exceptions import BadRequest
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from util import token_required
from flask_cors import CORS
import io

IMAGE_SIZE = 192
# Load your trained model1
mobile_net_model = tf.keras.models.load_model('mobile_net_debit.keras')
authorizations = {
    'apikey': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'X-API-KEY'
    }
}

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'c8843ce9-57c7-4042-8e66-9692331876d3'
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app, title="Image Recognition API", description="This Webservice is built around Tensorflow Models used "
                                                          "for Image Classification and provides simple endpoints "
                                                          "to use them", contact_email="lukas.burtscher@check24.at",
          authorizations=authorizations
          )

request = api.model('Picture', {
    'image': fields.String(required=True, description='Base64 encoded String of PNG or JPG File')
})

response = api.model('Response', {
    'accepted': fields.Boolean(default=True),
    'confidence': fields.Float()
})

ns = api.namespace('Recognition API', description='Visual Classifier', path="/api/recognitions")


@ns.route('/debit-card')
class DebitCard(Resource):
    '''POST a base64 encoded picture string to get prediction of the Picture'''

    @ns.doc('process image', security="apikey")
    @ns.expect(request)
    @ns.marshal_with(response, code=200)
    @token_required
    def post(self):
        data = api.payload
        image_b64 = data['image']
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
            prob_class_1 = 1 / (1 + np.exp(-predictions))
            prob_class_0 = 1 - prob_class_1
            p_array = [prob_class_0[0], prob_class_1[0]]
            predictions = tf.where(score < 0.5, 0, 1)[0]
            p = predictions.numpy()[0]
            return {"accepted": bool(p == 0), "confidence": round(float(p_array[p]), 4)}, 200

        except Exception as e:
            raise BadRequest(description=str(e) + "(File probably not PNG or JPG)")


if __name__ == '__main__':
    app.run(debug=True, port=8080)
