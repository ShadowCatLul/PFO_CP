FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    wget \
    gnupg2 \
    software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Копируем файл requirements.txt в контейнер
COPY requirements.txt /workspace/requirements.txt

# Установка зависимостей из requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Проверка установки
# RUN python -c "import torch; print(torch.cuda.is_available())"

# Установим рабочую директорию
WORKDIR /workspace

# По умолчанию запускаем bash
CMD ["/bin/bash"]