FROM python:3.11.4-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
ENV GIT_PYTHON_REFRESH=quiet
WORKDIR /app
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
COPY . .
EXPOSE 8050
USER 10001
CMD ["python", "main.py"]
WORKDIR /app
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
COPY . .
EXPOSE 8050
USER 10001
CMD ["python", "main.py"]
