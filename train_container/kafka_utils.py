from kafka import KafkaProducer, KafkaConsumer # type: ignore
import json
import os
import time

def create_producer():
    """
    Creates a KafkaProducer configured to send JSON-serialized messages.
    """
    bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS")
    if not bootstrap_servers:
        raise ValueError("KAFKA_BOOTSTRAP_SERVERS environment variable not set.")
        
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        # Serialize Python dicts to a JSON string, then encode to UTF-8 bytes.
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        # Keys are often strings, encode them to bytes.
        key_serializer=lambda k: k.encode('utf-8') if k else None
    )
    return producer

def create_consumer(topic, group_id):
    """
    Creates a KafkaConsumer configured to receive and deserialize JSON messages.

    Args:
        topic (str): The topic to subscribe to.
        group_id (str): The consumer group ID for this consumer.

    Returns:
        KafkaConsumer: A configured consumer instance ready to poll messages.
    """
    bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS')
    if not bootstrap_servers:
        raise ValueError("KAFKA_BOOTSTRAP_SERVERS environment variable not set.")

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        key_deserializer=lambda k: k.decode('utf-8') if k else None,
        auto_offset_reset='earliest',
        enable_auto_commit=False,  # manual commit to support retry / DLQ logic
        max_poll_records=10,
    )
    return consumer

def produce_message(producer, topic, value, key=None):
    """
    Sends a dictionary as a JSON message to a Kafka topic.

    Args:
        producer (KafkaProducer): The producer instance to use.
        topic (str): The target topic name.
        value (dict): The dictionary payload to be sent as JSON.
        key (str, optional): The message key. Defaults to None.
    """
    try:
        producer.send(topic, value=value, key=key)
        producer.flush()
        print(f"Successfully sent JSON message with key '{key}' to topic '{topic}'.")
        return True
    except Exception as e:
        print(f"Error sending message to Kafka: {e}")

def consume_messages(consumer, callback):
    """
    Continuously listens for messages and processes them using a callback function.

    Args:
        consumer (KafkaConsumer): The consumer instance to use.
        callback (function): A function to be called for each message. 
                             It should accept one argument: the message.
    """
    print("Consumer loop started. Waiting for messages...")
    try:
        for message in consumer:
            try:
                callback(message)
            except Exception as e:
                print(f"Error processing message: {message.value}")
                print(f"Error details: {e}")
    except KeyboardInterrupt:
        print("Consumer process interrupted by user.")
    finally:
        print("Closing Kafka consumer.")
        consumer.close()

def publish_error(producer, dlq_topic, operation, status, error_details, payload):
    """
    Constructs an error message and sens it to the DLQ
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