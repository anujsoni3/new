# auth.py

import os
from flask import request, jsonify
from functools import wraps
from dotenv import load_dotenv

load_dotenv()
AUTHORIZED_TOKEN = os.getenv("BACKEND_BEARER_TOKEN")

def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header."}), 401

        token = auth_header.split(" ")[1]
        if token != AUTHORIZED_TOKEN:
            return jsonify({"error": "Unauthorized"}), 403

        return f(*args, **kwargs)
    return wrapper
