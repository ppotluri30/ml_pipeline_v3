"""
Production-grade Kafka worker for FLTS ML Pipeline.

Consumes from Kafka topics and handles:
- Model promotion events (model-selected)
- Training completion notifications (model-training)
- Batch inference requests (inference-data)

Features:
- Graceful shutdown on SIGTERM/SIGINT
- Backpressure control to prevent CPU saturation
- Exponential backoff on errors
- Safe offset commits
- Reconnection logic
- Memory-efficient message processing
"""
import os
import sys
import time
import json as _json
import signal
import threading
from typing import Optional

# Ensure shared modules are available
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

from kafka_utils import create_producer, create_consumer_configurable
from client_utils import post_file

# Environment configuration
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://fastapi-app:8000")
IDENTIFIER = os.environ.get("IDENTIFIER", "default")
CONSUMER_GROUP_ID = os.environ.get("CONSUMER_GROUP_ID", "inference-worker")

# Kafka topics
INFERENCE_DATA_TOPIC = os.environ.get("CONSUMER_TOPIC_0", "inference-data")
MODEL_TRAINING_TOPIC = os.environ.get("CONSUMER_TOPIC_1", "model-training")
PROMOTION_TOPIC = os.environ.get("PROMOTION_TOPIC", "model-selected")

# Consumer configuration with backpressure
USE_MANUAL_COMMIT = os.environ.get("USE_MANUAL_COMMIT", "true").lower() == "true"
FETCH_MAX_WAIT_MS = int(os.environ.get("FETCH_MAX_WAIT_MS", "500"))  # Increased from 50
MAX_POLL_RECORDS = int(os.environ.get("MAX_POLL_RECORDS", "10"))  # Reduced from 64
POLL_TIMEOUT_MS = int(os.environ.get("POLL_TIMEOUT_MS", "1000"))  # 1 second poll
IDLE_SLEEP_SECONDS = float(os.environ.get("IDLE_SLEEP_SECONDS", "0.5"))  # Sleep when no messages

# Graceful shutdown flag
shutdown_requested = threading.Event()


def _log(event: str, **kwargs):
    """Structured logging."""
    log_entry = {"service": "inference_worker", "event": event}
    log_entry.update(kwargs)
    print(_json.dumps(log_entry), flush=True)


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    _log("shutdown_signal_received", signal=signum)
    shutdown_requested.set()


# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def _handle_promotion_message(message, producer) -> bool:
    """
    Handle model-selected events by writing promotion pointer.
    
    Returns True if successfully processed, False otherwise.
    """
    try:
        # Handle both dict and string message values
        if isinstance(message.value, dict):
            data = message.value
        elif isinstance(message.value, (str, bytes)):
            data = _json.loads(message.value)
        else:
            _log("promotion_message_invalid_type", type=str(type(message.value)))
            return False
        
        run_id = data.get("run_id")
        model_uri = data.get("model_uri")
        model_type = data.get("model_type")
        config_hash = data.get("config_hash", "unknown")
        
        if not all([run_id, model_uri, model_type]):
            _log("promotion_message_invalid", data=data)
            return False
        
        _log("promotion_received", run_id=run_id, model_type=model_type, config_hash=config_hash)
        
        # Create promotion pointer
        import datetime
        pointer = {
            "run_id": run_id,
            "model_uri": model_uri,
            "model_type": model_type,
            "config_hash": config_hash,
            "promoted_at": datetime.datetime.utcnow().isoformat() + "Z",
            "identifier": data.get("identifier", IDENTIFIER),
        }
        
        # Write to root-level current.json (global pointer)
        pointer_data = _json.dumps(pointer, indent=2).encode('utf-8')
        
        # Retry logic for pointer write
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                post_file(GATEWAY_URL, "model-promotion", "current.json", pointer_data)
                _log("promotion_pointer_written", run_id=run_id, model_type=model_type, 
                     attempt=attempt, path="model-promotion/current.json")
                return True
            except Exception as write_err:
                _log("promotion_pointer_write_failed", run_id=run_id, attempt=attempt,
                     error=str(write_err))
                if attempt < max_retries:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return False
        
    except Exception as e:
        _log("promotion_handler_error", error=str(e), message_value=str(message.value)[:200])
        return False


def _handle_model_training_message(message, producer) -> bool:
    """Handle model-training completion notifications."""
    try:
        # Handle both dict and string message values
        if isinstance(message.value, dict):
            data = message.value
        else:
            data = _json.loads(message.value)
        
        operation = data.get("operation", "")
        status = data.get("status", "")
        run_id = data.get("run_id", "")
        config_hash = data.get("config_hash", "unknown")
        
        _log("training_notification", operation=operation, status=status, 
             run_id=run_id, config_hash=config_hash)
        return True
        
    except Exception as e:
        _log("training_handler_error", error=str(e))
        return False


def _handle_inference_data_message(message, producer) -> bool:
    """Handle batch inference requests (future feature)."""
    try:
        # Handle both dict and string message values
        if isinstance(message.value, dict):
            data = message.value
        else:
            data = _json.loads(message.value)
        
        _log("inference_data_received", bucket=data.get("bucket"), 
             object_key=data.get("object_key"))
        # TODO: Implement batch inference processing
        return True
        
    except Exception as e:
        _log("inference_data_handler_error", error=str(e))
        return False


def _consumer_loop(topic_name: str, handler_func, producer, consumer_group_id: str):
    """
    Production-grade Kafka consumer loop with:
    - Graceful shutdown
    - Backpressure control
    - Exponential backoff on errors
    - Safe commit logic
    """
    consumer = None
    error_count = 0
    max_errors = 10
    
    _log("consumer_loop_starting", topic=topic_name, group=consumer_group_id)
    
    while not shutdown_requested.is_set():
        try:
            # Create or recreate consumer
            if consumer is None:
                consumer = create_consumer_configurable(
                    topic_name,
                    consumer_group_id,
                    auto_offset_reset='earliest',
                    enable_auto_commit=not USE_MANUAL_COMMIT,
                    max_poll_records=MAX_POLL_RECORDS,
                    fetch_max_wait_ms=FETCH_MAX_WAIT_MS,
                )
                _log("consumer_created", topic=topic_name, group=consumer_group_id,
                     manual_commit=USE_MANUAL_COMMIT)
                error_count = 0  # Reset error count on successful connection
            
            # Poll for messages with timeout
            records = consumer.poll(timeout_ms=POLL_TIMEOUT_MS, max_records=MAX_POLL_RECORDS)
            
            if not records:
                # No messages - apply backpressure by sleeping
                if not shutdown_requested.is_set():
                    time.sleep(IDLE_SLEEP_SECONDS)
                continue
            
            # Process messages
            processed_count = 0
            failed_count = 0
            
            for tp, messages in records.items():
                for msg in messages:
                    if shutdown_requested.is_set():
                        break
                    
                    try:
                        success = handler_func(msg, producer)
                        if success:
                            processed_count += 1
                        else:
                            failed_count += 1
                    except Exception as handler_err:
                        _log("message_handler_exception", topic=topic_name, 
                             partition=msg.partition, offset=msg.offset,
                             error=str(handler_err))
                        failed_count += 1
            
            # Commit offsets if manual commit enabled
            if USE_MANUAL_COMMIT and records:
                try:
                    consumer.commit()
                    _log("offsets_committed", topic=topic_name, 
                         processed=processed_count, failed=failed_count)
                except Exception as commit_err:
                    _log("commit_error", topic=topic_name, error=str(commit_err))
            
            # Log processing batch
            if processed_count > 0 or failed_count > 0:
                _log("batch_processed", topic=topic_name, 
                     processed=processed_count, failed=failed_count)
        
        except Exception as e:
            error_count += 1
            _log("consumer_loop_error", topic=topic_name, error=str(e), 
                 error_count=error_count)
            
            # Close and recreate consumer on error
            if consumer:
                try:
                    consumer.close()
                except:
                    pass
                consumer = None
            
            # Exponential backoff
            if error_count >= max_errors:
                _log("consumer_max_errors_reached", topic=topic_name)
                break
            
            backoff_time = min(2 ** error_count, 60)  # Max 60 seconds
            _log("consumer_backoff", topic=topic_name, seconds=backoff_time)
            time.sleep(backoff_time)
    
    # Cleanup
    _log("consumer_loop_shutting_down", topic=topic_name)
    if consumer:
        try:
            consumer.close()
            _log("consumer_closed", topic=topic_name)
        except Exception as close_err:
            _log("consumer_close_error", topic=topic_name, error=str(close_err))


def start_kafka_worker():
    """Start Kafka consumer workers with production-grade settings."""
    _log("kafka_worker_starting", group_id=CONSUMER_GROUP_ID)
    
    # Create healthcheck file for probes
    try:
        with open('/tmp/worker-healthy', 'w') as f:
            f.write('started')
    except Exception as health_err:
        _log("healthcheck_file_error", error=str(health_err))
    
    # Create Kafka producer
    producer = None
    try:
        producer = create_producer()
        _log("kafka_producer_created")
    except Exception as prod_err:
        _log("kafka_producer_error", error=str(prod_err))
        # Continue without producer - some handlers can work without it
    
    # Start consumer threads (non-daemon to allow graceful shutdown)
    threads = []
    
    # Consumer for model-selected (promotion) - CRITICAL
    promotion_thread = threading.Thread(
        target=_consumer_loop,
        args=(PROMOTION_TOPIC, _handle_promotion_message, producer, CONSUMER_GROUP_ID),
        daemon=False,  # Non-daemon for graceful shutdown
        name="promotion-consumer"
    )
    promotion_thread.start()
    threads.append(promotion_thread)
    _log("consumer_thread_started", topic=PROMOTION_TOPIC, thread=promotion_thread.name)
    
    # Consumer for model-training (notifications)
    training_thread = threading.Thread(
        target=_consumer_loop,
        args=(MODEL_TRAINING_TOPIC, _handle_model_training_message, producer, CONSUMER_GROUP_ID),
        daemon=False,
        name="training-consumer"
    )
    training_thread.start()
    threads.append(training_thread)
    _log("consumer_thread_started", topic=MODEL_TRAINING_TOPIC, thread=training_thread.name)
    
    # Consumer for inference-data (batch inference)
    inference_thread = threading.Thread(
        target=_consumer_loop,
        args=(INFERENCE_DATA_TOPIC, _handle_inference_data_message, producer, CONSUMER_GROUP_ID),
        daemon=False,
        name="inference-consumer"
    )
    inference_thread.start()
    threads.append(inference_thread)
    _log("consumer_thread_started", topic=INFERENCE_DATA_TOPIC, thread=inference_thread.name)
    
    _log("kafka_worker_ready", topics=[PROMOTION_TOPIC, MODEL_TRAINING_TOPIC, INFERENCE_DATA_TOPIC],
         thread_count=len(threads))
    
    # Main loop - monitor threads and handle shutdown
    try:
        while not shutdown_requested.is_set():
            time.sleep(5)  # Check every 5 seconds
            
            # Check thread health
            for t in threads:
                if not t.is_alive():
                    _log("consumer_thread_died", thread=t.name)
            
            # Heartbeat log every 30 seconds
            if int(time.time()) % 30 == 0:
                alive_threads = [t.name for t in threads if t.is_alive()]
                _log("worker_heartbeat", alive_threads=alive_threads)
                time.sleep(1)  # Avoid duplicate heartbeats
    
    except KeyboardInterrupt:
        _log("kafka_worker_interrupted")
        shutdown_requested.set()
    
    # Wait for threads to finish gracefully
    _log("kafka_worker_shutting_down", waiting_for_threads=len(threads))
    for t in threads:
        t.join(timeout=10)
        if t.is_alive():
            _log("thread_shutdown_timeout", thread=t.name)
    
    # Close producer
    if producer:
        try:
            producer.close(timeout=5)
            _log("kafka_producer_closed")
        except Exception as close_err:
            _log("producer_close_error", error=str(close_err))
    
    # Remove healthcheck file
    try:
        os.remove('/tmp/worker-healthy')
    except:
        pass
    
    _log("kafka_worker_stopped")


if __name__ == "__main__":
    try:
        start_kafka_worker()
    except Exception as e:
        _log("kafka_worker_fatal_error", error=str(e))
        sys.exit(1)
