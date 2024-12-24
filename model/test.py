# Пример отправки POST-запроса с изображением в формате multipart/form-data
import requests

url = "http://localhost:5000/upload"
image_path = "G:\\photo_2024-02-10_21-17-43.jpg"

files = {'file': open(image_path, 'rb')}

response = requests.post(url, files=files)
print(response.json())
