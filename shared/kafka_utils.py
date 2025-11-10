from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

from kafka import KafkaConsumer, KafkaProducer  # type: ignore


def _require_bootstrap_servers() -> str:
    bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS")
    if not bootstrap_servers:
        raise ValueError("KAFKA_BOOTSTRAP_SERVERS environment variable not set.")
    return bootstrap_servers


def create_producer(**overrides: Any) -> KafkaProducer:
    """
    Create a KafkaProducer configured to send JSON-serialized messages.
    Optional keyword overrides are forwarded to the KafkaProducer constructor.
    """
    bootstrap_servers = overrides.pop("bootstrap_servers", None) or _require_bootstrap_servers()

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        security_protocol="PLAINTEXT",
        api_version=(2, 5, 0),
        **overrides,
    )
    return producer


def create_consumer(topic: str, group_id: str, **overrides: Any) -> KafkaConsumer:
    """
    Create a KafkaConsumer configured to receive and deserialize JSON messages.
    """
    bootstrap_servers = overrides.pop("bootstrap_servers", None) or _require_bootstrap_servers()

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        auto_offset_reset=overrides.pop("auto_offset_reset", "earliest"),
        security_protocol="PLAINTEXT",
        api_version=(2, 5, 0),
        **overrides,
    )
    return consumer


def create_consumer_configurable(
    topic: str,
    group_id: str,
    *,
    enable_auto_commit: bool = False,
    auto_offset_reset: str = "earliest",
    **overrides: Any,
) -> KafkaConsumer:
    """
    Create a KafkaConsumer with explicit control over auto-commit behaviour.

    Poll batching parameters must be passed to consumer.poll(...).
    """
    bootstrap_servers = overrides.pop("bootstrap_servers", None) or _require_bootstrap_servers()

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        auto_offset_reset=auto_offset_reset,
        enable_auto_commit=enable_auto_commit,
        security_protocol="PLAINTEXT",
        api_version=(2, 5, 0),
        **overrides,
    )
    return consumer


def produce_message(
    producer: KafkaProducer,
    topic: str,
    value: Dict[str, Any],
    key: str | None = None,
) -> bool:
    """
    Sends a dictionary as a JSON message to a Kafka topic.
    """
    try:
        producer.send(topic, value=value, key=key)
        producer.flush()
        print(f"Successfully sent JSON message with key '{key}' to topic '{topic}'.")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"Error sending message to Kafka: {exc}")
        return False


def consume_messages(consumer: KafkaConsumer, callback) -> None:
    """
    Continuously listens for messages and processes them using a callback function.
    """
    print("Consumer loop started. Waiting for messages...")
    try:
        for message in consumer:
            try:
                callback(message)
            except Exception as exc:  # noqa: BLE001
                print(f"Error processing message: {message.value}")
                print(f"Error details: {exc}")
    except KeyboardInterrupt:
        print("Consumer process interrupted by user.")
    finally:
        print("Closing Kafka consumer.")
        consumer.close()


def jlog(event: str, **extra: Any) -> None:
    try:
        base = {"service": "kafka_utils", "event": event}
        base.update({k: v for k, v in extra.items() if v is not None})
        print(json.dumps(base))
    except Exception:  # noqa: BLE001
        pass


def commit_offsets_sync(consumer: KafkaConsumer, offsets_by_tp: Dict[Any, Any]) -> None:
    """
    Helper to commit offsets synchronously.

    offsets_by_tp should be a dict of {TopicPartition: OffsetAndMetadata}.
    """
    try:
        consumer.commit(offsets=offsets_by_tp)
        jlog(
            "commit_offsets",
            tps=[f"{tp.topic}:{tp.partition}" for tp in offsets_by_tp.keys()],
            offsets=[o.offset for o in offsets_by_tp.values()],
        )
    except Exception as exc:  # noqa: BLE001
        jlog("commit_offsets_fail", error=str(exc))


def publish_error(
    producer: KafkaProducer,
    dlq_topic: str,
    operation: str,
    status: str,
    error_details: str,
    payload: Dict[str, Any],
) -> None:
    """
    Constructs an error message and sends it to the DLQ.
    """
    error_message = {
        "operation": operation,
        "status": status,
        "error": error_details,
        "payload": payload,
        "timestamp": time.time(),
    }
    print(f"Sending error to DLQ: {error_message}")
    produce_message(producer, dlq_topic, error_message)



