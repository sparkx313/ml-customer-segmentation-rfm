FROM python:3.11-slim
WORKDIR /app
COPY . .
CMD ["python", "rfm_segmentation.py"]
