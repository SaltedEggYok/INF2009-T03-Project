import time
import paho.mqtt.client as mqtt
import base64

def on_publish(client, userdata, mid, reason_code, properties):
	print(f"Message {mid} published.")

#unacked_publish = set()
mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqttc.on_publish = on_publish

#mqttc.user_data_set(unacked_publish)
mqttc.connect("192.168.26.142")
mqttc.loop_start()

# Path to your emotion_dict.txt file
file_path = '/home/houwei/Lab2/emotion_dict.txt'

# Read and encode the file
with open(file_path, "rb") as file:
    file_content = file.read()
    encoded_content = base64.b64encode(file_content)

# Publish the encoded file content
topic = "emotion/face"
mqttc.publish(topic, payload=encoded_content, qos=1)

mqttc.disconnect()
mqttc.loop_stop()
