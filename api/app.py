from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/")
def home():
    data = {
        "message": "Hello, World!",
        "status": "success"
    }
    return jsonify(data)

@app.route("/about")
def about():
    return "About"
