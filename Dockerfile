FROM python:3.12-slim
WORKDIR /srv/narezka
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py .
ENV PORT=8788 PYTHONUNBUFFERED=1
EXPOSE 8788
VOLUME ["/srv/narezka/clips"]
CMD ["python", "server.py"]
