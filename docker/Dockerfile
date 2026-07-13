FROM python:3.11-slim

WORKDIR /app

# CPU-only torch — CUDA build would be several GB and useless on a free-tier host anyway
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/

ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]