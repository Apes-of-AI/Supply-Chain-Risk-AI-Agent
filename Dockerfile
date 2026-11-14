# Optional: only needed if you switch the Space to the Docker SDK, or if you
# want the same image inside an enclave where you control egress.
FROM python:3.12-slim

RUN useradd -m -u 1000 app
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app . .
USER app

ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 7860
CMD ["python", "app.py"]
