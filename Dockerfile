FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY rfm_segmentation.py .
ENTRYPOINT ["python", "rfm_segmentation.py"]