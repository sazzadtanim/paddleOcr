FROM python:3.10-slim

WORKDIR /app

# System deps for OpenCV/Paddle
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the PP-OCRv6 English models (~150MB) into the image so the first
# request after deploy is fast and the container needs no runtime network
# access. Placed BEFORE `COPY app.py` so editing app.py does not invalidate this
# layer (no re-download on every app change). Mirrors the exact config in app.py.
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(ocr_version='PP-OCRv6', lang='en', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=True)"

COPY app.py .
COPY static/ ./static/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
