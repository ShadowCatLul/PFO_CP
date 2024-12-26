# Frontend (Flask application for handling RGB and predictions)

from flask import Flask, render_template, request, jsonify
import rasterio
import numpy as np
from PIL import Image
import io
import base64
import requests
# Initialize Flask application
app = Flask(__name__)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/show_rgb', methods=['POST'])
def show_rgb():
    """Handle RGB visualization requests."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        try:
            rgb_image = generate_rgb_image(file)
            img_io = io.BytesIO()
            rgb_image.save(img_io, 'PNG')
            img_io.seek(0)
            img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
            return jsonify({"rgb_image": f"data:image/png;base64,{img_base64}"})
        except Exception as e:
            return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        # Отправляем изображение в бэкенд
        backend_url = "http://backend:5001/process"
        response = requests.post(backend_url, files={'file': file})
        if response.status_code == 200:
            data = response.json()
            return jsonify(data)
        else:
            return jsonify({"error": "Failed to process image in backend."}), 500


def generate_rgb_image(file):
    """Generate an RGB image from a multispectral TIF file."""
    with rasterio.open(file) as src:
        blue = src.read(1)
        green = src.read(2)
        red = src.read(3)
        rgb = np.stack((red, green, blue), axis=-1)
        rgb_image = Image.fromarray((rgb / rgb.max() * 255).astype('uint8'))
        return rgb_image

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)