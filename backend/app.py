from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)
MODEL_CONTAINER_URL = "http://model:5002/predict"

@app.route('/process', methods=['POST'])
def process():
    """Handle requests from frontend and forward them to the model container."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file:
        try:
            # Отправляем изображение в модельный контейнер
            response = requests.post(MODEL_CONTAINER_URL, files={'file': file.read()})
            print(response.status_code)
            print(response)
            if response.status_code == 200:
                return jsonify(response.json())
            else:
                return jsonify({"error": "Failed to process image in model container."}), 500
        except Exception as e:
            return jsonify({"error": f"Backend error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)