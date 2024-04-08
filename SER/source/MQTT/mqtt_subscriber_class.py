import time
import paho.mqtt.client as mqtt
import base64

# Using your pi as broker, connect to your pi's ip address; change the ip address below accordingly
# BROKER = "192.168.26.142"
# Alternatively, I think can use the cloud as broker to test the code; uncomment the line below
BROKER = "mqtt.eclipseprojects.io"

class MQTTSubscriber:
    def __init__(self, broker_address="mqtt.eclipseprojects.io", client_id=""):
        '''
        Constructor
        :param broker_address: address of the broker
        :type broker_address: str
        :param client_id: client id
        :type client_id: str
        '''
        # self.client = mqtt.Client(client_id if client_id else mqtt.base62(uuid.uuid4().int, padding=22), mqtt.CallbackAPIVersion.VERSION_5)
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_subscribe = self.on_subscribe
        self.client.on_unsubscribe = self.on_unsubscribe
        self.broker_address = broker_address

    def on_connect(self, client, userdata, flags, reason_code, properties):
        '''
        Function to handle the on_connect event
        :param client: mqtt client
        :type client: mqtt.Client
        :param userdata: user data
        :type userdata: Any
        :param flags: flags
        :type flags: dict
        :param reason_code: reason code
        :type reason_code: int
        :param properties: properties
        :type properties: MQTTProperties
        :return: None
        '''
        if reason_code.is_failure:
            print(f"Failed to connect: {reason_code}. loop_forever() will retry connection")
        else:
            print("Connected successfully. Waiting for subscription commands.")

    def on_message(self, client, userdata, message):
        '''
        Function to handle the on_message event
        :param client: mqtt client
        :type client: mqtt.Client
        :param userdata: user data
        :type userdata: Any
        :param message: message
        :type message: MQTTMessage
        :return: None
        '''
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
        '''
        Function to handle the on_subscribe event
        :param client: mqtt client
        :type client: mqtt.Client
        :param userdata: user data
        :type userdata: Any
        :param mid: message id
        :type mid: int
        :param reason_code_list: reason code list
        :type reason_code_list: list
        :param properties: properties
        :type properties: MQTTProperties
        :return: None    
        '''
        # Since we subscribed only for a single channel, reason_code_list contains
        # a single entry
        if reason_code_list[0].is_failure:
            print(f"Broker rejected you subscription: {reason_code_list[0]}")
        else:
            print(f"Subscription acknowledged. Broker granted the following QoS: {reason_code_list[0].value}")

    def on_unsubscribe(self, client, userdata, mid, reason_code_list, properties):
        '''
        Function to handle the on_unsubscribe event
        :param client: mqtt client
        :type client: mqtt.Client
        :param userdata: user data
        :type userdata: Any
        :param mid: message id
        :type mid: int
        :param reason_code_list: reason code list
        :type reason_code_list: list
        :param properties: properties
        :type properties: MQTTProperties
        :return: None
        '''
        print("Unsubscription acknowledged.")

    def connect(self):
        '''
        Function to connect to the MQTT broker
        :param None
        :return: None
        '''
        self.client.connect(self.broker_address)
        self.client.loop_start()

    def subscribe(self, topics):
        '''
        Function to subscribe to topics
        :param topics: list of topics to subscribe to
        :type topics: list
        :return: None
        '''
        self.client.subscribe(topics)
        print(f"Subscribed to topics: {topics}")

    def unsubscribe(self, topics):
        '''
        Function to unsubscribe from topics
        :param topics: list of topics to unsubscribe from
        :type topics: list
        :return: None
        '''
        self.client.unsubscribe(topics)
        print(f"Unsubscribed from topics: {topics}")

    def disconnect(self):
        '''
        Function to disconnect from the MQTT broker
        :param None
        :return: None
        '''
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
