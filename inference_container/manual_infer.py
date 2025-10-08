import os, io
import pyarrow.parquet as pq
from inferencer import Inferencer
from kafka_utils import create_producer
from client_utils import get_file

GATEWAY_URL = os.environ['GATEWAY_URL']
EXPERIMENT = os.environ.get('MANUAL_EXPERIMENT','Default')
RUN_NAME = os.environ.get('MANUAL_RUN','GRU')
IDENTIFIER = os.environ.get('IDENTIFIER','manualtest')

print(f"[manual_infer] Starting manual inference for experiment={EXPERIMENT} run={RUN_NAME} identifier={IDENTIFIER}")
producer = create_producer()
inf = Inferencer(GATEWAY_URL, producer, 'DLQ-performance-eval','performance-eval')
inf.load_model(EXPERIMENT, RUN_NAME)

print("[manual_infer] Downloading evaluation parquet")
parquet_buf = get_file(GATEWAY_URL,'processed-data','test_processed_data.parquet')
parquet_buf.seek(0,2)
size = parquet_buf.tell()
parquet_buf.seek(0)
print(f"[manual_infer] Parquet bytes: {size}")

table = pq.read_table(source=parquet_buf)
df = table.to_pandas()
print(f"[manual_infer] DataFrame shape: {df.shape}")

print("[manual_infer] Running perform_inference()")
inf.perform_inference(df)
print("[manual_infer] Done.")
