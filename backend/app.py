from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)
MODEL_CONTAINER_URL = "http://model:5002/predict"

@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file:
        response = requests.post(MODEL_CONTAINER_URL, files={'file': file.read()})
        return jsonify(response.json())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)