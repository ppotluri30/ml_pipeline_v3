# mlflow_container/Dockerfile
FROM ghcr.io/mlflow/mlflow:v3.3.1

# Install psycopg2 (needed for Postgres backend)
RUN pip install "mlflow[extras]==3.3.1" psycopg2-binary awscli awscli-local
