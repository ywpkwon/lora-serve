
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -e .
EXPOSE 8000
CMD ["uvicorn", "lora_serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
