import time
import paho.mqtt.client as mqtt
import base64

# Using your pi as broker, connect to your pi's ip address; change the ip address below accordingly
# BROKER = "192.168.26.142"
# Alternatively, I think can use the cloud as broker to test the code; uncomment the line below
BROKER = "mqtt.eclipseprojects.io"

class MQTTPublisher:
    def __init__(self, broker_address=BROKER, client_id=""):
        self.broker_address = broker_address
        # self.client = mqtt.Client(client_id if client_id else mqtt.base62(uuid.uuid4().int, padding=22), mqtt.CallbackAPIVersion.VERSION_5)
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_publish = self.on_publish

    def on_publish(self, client, userdata, mid, reason_code, properties=None):
        print(f"Message {mid} published.")

    def connect(self):
        self.client.connect(self.broker_address)
        self.client.loop_start()

    def publish_payload(self, topic, payload, qos=2):
        # This method assumes payload is already encoded
        self.client.publish(topic, payload=payload, qos=qos)

    def disconnect(self):
        self.client.disconnect()
        self.client.loop_stop()

    @staticmethod
    def encode_file_to_base64(file_path):
        with open(file_path, "rb") as file:
            file_content = file.read()
            return base64.b64encode(file_content)

# Test Usage
if __name__ == "__main__":
    file_path = 'test_payload.txt'
    topic = "emotion/face"
    # topic = "emotion/speech"
    
    # Initialize MQTTPublisher
    publisher = MQTTPublisher()

    # Connect to the MQTT broker
    publisher.connect()
    
    # Wait a bit for the client to establish connection
    time.sleep(1)
    
    # Encode file content
    encoded_content = MQTTPublisher.encode_file_to_base64(file_path)

    # Publish the encoded file content
    publisher.publish_payload(topic, payload=encoded_content, qos=2)
    
    # Wait a bit to ensure the message is published before disconnecting
    time.sleep(1)

    # Disconnect cleanly
    publisher.disconnect()
