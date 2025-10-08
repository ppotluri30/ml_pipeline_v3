import mlflow
import os

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
experiments = mlflow.search_experiments()
print("Experiments:", [e.name for e in experiments])
