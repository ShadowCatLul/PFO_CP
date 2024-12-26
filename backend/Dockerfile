FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл requirements.txt в контейнер
COPY requirements.txt /workspace/requirements.txt

# Установка зависимостей из requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Проверка установки
# RUN python -c "import torch; print(torch.cuda.is_available())"
COPY . /workspace
# Установим рабочую директорию
WORKDIR /workspace

# По умолчанию запускаем Flask-приложение
CMD ["flask", "run", "--host=0.0.0.0", "--port=5001"]