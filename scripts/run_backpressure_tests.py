import os, sys, types
sys.path.insert(0, os.getcwd())
# Env for tests
os.environ.setdefault('DISABLE_BUCKET_ENSURE','1')
os.environ.setdefault('INFERENCE_AUTOSTART','0')
os.environ.setdefault('USE_BOUNDED_QUEUE','1')
os.environ.setdefault('QUEUE_MAXSIZE','32')
os.environ.setdefault('ENABLE_MICROBATCH','1')
os.environ.setdefault('BATCH_SIZE','8')
os.environ.setdefault('BATCH_TIMEOUT_MS','5')
os.environ.setdefault('GATEWAY_URL','http://localhost:8000')
os.environ.setdefault('CONSUMER_TOPIC_0','inference-data')
os.environ.setdefault('CONSUMER_TOPIC_1','model-training')
os.environ.setdefault('PROMOTION_TOPIC','model-selected')
os.environ.setdefault('CONSUMER_GROUP_ID','test-group')
os.environ.setdefault('PRODUCER_TOPIC','performance-eval')
os.environ.setdefault('KAFKA_BOOTSTRAP_SERVERS','localhost:9092')

# Mock heavy modules (leave pandas to a custom stub below)
for mod in ['pyarrow','pyarrow.parquet','mlflow']:
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)

# Minimal torch stub with device and cuda.is_available
if 'torch' not in sys.modules:
    tmod = types.ModuleType('torch')
    class _Cuda:
        @staticmethod
        def is_available():
            return False
    def device(name):
        return name
    tmod.cuda = _Cuda()
    tmod.device = device
    sys.modules['torch'] = tmod

# Provide/augment a minimal pandas stub with Series/DataFrame/Timedelta
if 'pandas' not in sys.modules:
    pdmod = types.ModuleType('pandas')
    class Series: pass
    class DataFrame: pass
    class Timedelta:
        def __init__(self, minutes=1, **kwargs):
            self._minutes = minutes
    pdmod.Series = Series
    pdmod.DataFrame = DataFrame
    pdmod.Timedelta = Timedelta
    sys.modules['pandas'] = pdmod
else:
    pdmod = sys.modules['pandas']
    if not hasattr(pdmod, 'Series'):
        class Series: pass
        pdmod.Series = Series
    if not hasattr(pdmod, 'DataFrame'):
        class DataFrame: pass
        pdmod.DataFrame = DataFrame
    if not hasattr(pdmod, 'Timedelta'):
        class Timedelta:
            def __init__(self, minutes=1, **kwargs):
                self._minutes = minutes
        pdmod.Timedelta = Timedelta

from inference_container.tests.test_backpressure import (
    test_bounded_queue_and_backpressure, test_ttl_expiry, test_microbatch_drain
)

print('Running test_bounded_queue_and_backpressure...')
test_bounded_queue_and_backpressure()
print('PASS: bounded queue')

print('Running test_ttl_expiry...')
test_ttl_expiry()
print('PASS: ttl expiry')

print('Running test_microbatch_drain...')
test_microbatch_drain()
print('PASS: microbatch drain')

print('All tests passed.')
