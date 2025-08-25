FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    poppler-utils \
    build-essential \
    gcc \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libfreetype6-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Installer torch séparément (optimisation cache)
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY ./app ./app

EXPOSE 5000
