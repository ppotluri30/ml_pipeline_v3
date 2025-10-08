from kafka import KafkaProducer, KafkaConsumer  # type: ignore
import json
import os


def create_producer():
    bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS")
    if not bootstrap_servers:
        raise ValueError("KAFKA_BOOTSTRAP_SERVERS environment variable not set.")
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
    )


def create_consumer(topic, group_id):
    bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS")
    if not bootstrap_servers:
        raise ValueError("KAFKA_BOOTSTRAP_SERVERS environment variable not set.")
    return KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        max_poll_records=10,
    )


def produce_message(producer, topic, value, key=None):
    producer.send(topic, value=value, key=key)
    producer.flush()


def publish_error(producer, dlq_topic, operation, status, error_details, payload):
    error_message = {
        "operation": operation,
        "status": status,
        "error": error_details,
        "payload": payload,
    }
    produce_message(producer, dlq_topic, error_message)
