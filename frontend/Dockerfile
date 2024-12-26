FROM python:3.11-slim

# Установим системные зависимости для работы с изображениями
RUN apt-get update && apt-get install -y \
    wget \
    gnupg2 \
    software-properties-common \
    libgdal-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Копируем файл requirements.txt в контейнер
COPY requirements.txt /workspace/requirements.txt

COPY . /workspace

# Установка зависимостей из requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Установим рабочую директорию
WORKDIR /workspace

# По умолчанию запускаем Flask-приложение
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]