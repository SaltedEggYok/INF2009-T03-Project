import time
import paho.mqtt.client as mqtt
import base64

# Using your pi as broker, connect to your pi's ip address; change the ip address below accordingly
# BROKER = "192.168.26.142"
# Alternatively, I think can use the cloud as broker to test the code; uncomment the line below
BROKER = "mqtt.eclipseprojects.io"

class MQTTSubscriber:
    def __init__(self, broker_address="mqtt.eclipseprojects.io", client_id=""):
        # self.client = mqtt.Client(client_id if client_id else mqtt.base62(uuid.uuid4().int, padding=22), mqtt.CallbackAPIVersion.VERSION_5)
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_subscribe = self.on_subscribe
        self.client.on_unsubscribe = self.on_unsubscribe
        self.broker_address = broker_address

    def on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            print(f"Failed to connect: {reason_code}. loop_forever() will retry connection")
        else:
            print("Connected successfully. Waiting for subscription commands.")

    def on_message(self, client, userdata, message):
        print(f"Message received on topic {message.topic}")
        # Decode the message payload
        decoded_content = base64.b64decode(message.payload)
        # Path to save the file
        output_file_path = "output.txt"
        # Write the decoded content to a file
        with open(output_file_path, "wb") as output_file:
            output_file.write(decoded_content)
        print(f"File has been written to {output_file_path}")

    def on_subscribe(self, client, userdata, mid, reason_code_list, properties):
        # Since we subscribed only for a single channel, reason_code_list contains
        # a single entry
        if reason_code_list[0].is_failure:
            print(f"Broker rejected you subscription: {reason_code_list[0]}")
        else:
            print(f"Subscription acknowledged. Broker granted the following QoS: {reason_code_list[0].value}")

    def on_unsubscribe(self, client, userdata, mid, reason_code_list, properties):
        print("Unsubscription acknowledged.")

    def connect(self):
        self.client.connect(self.broker_address)
        self.client.loop_start()

    def subscribe(self, topics):
        self.client.subscribe(topics)
        print(f"Subscribed to topics: {topics}")

    def unsubscribe(self, topics):
        self.client.unsubscribe(topics)
        print(f"Unsubscribed from topics: {topics}")

    def disconnect(self):
        self.client.disconnect()
        self.client.loop_stop()

# Usage
if __name__ == "__main__":
    subscriber = MQTTSubscriber()
    subscriber.connect()
    time.sleep(1)  # Wait a bit for the connection to establish
    
    # Subscribe to topics
    topics = [("emotion/face", 2), ("emotion/voice", 2)]
    subscriber.subscribe(topics)
    
    # Keep the subscription alive for some time
    time.sleep(10)
    
    # Unsubscribe from topics (optional, demonstrate usage)
    subscriber.unsubscribe(["emotion/face", "emotion/voice"])
    
    # Disconnect after some time
    time.sleep(1)
    subscriber.disconnect()
