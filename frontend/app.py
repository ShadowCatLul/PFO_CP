from flask import Flask, render_template, request, jsonify
import requests
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        backend_url = "http://backend:5001/process"
        with open(filepath, 'rb') as f:
            response = requests.post(backend_url, files={'file': f})
        return jsonify(response.json())
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)