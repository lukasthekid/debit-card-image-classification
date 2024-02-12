from flask import request
from functools import wraps
from werkzeug.exceptions import Unauthorized
API_KEY = 'check24at'  # Replace with your actual API key


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if 'X-API-KEY' in request.headers:
            token = request.headers.get('X-API-KEY')

        if not token:
            raise Unauthorized(description="No API Key found")

        if token != API_KEY:
            raise Unauthorized(description="Wrong API Key")

        return f(*args, **kwargs)
    return decorated