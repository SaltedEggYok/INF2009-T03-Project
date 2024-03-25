import paho.mqtt.client as mqtt
import base64

def on_subscribe(client, userdata, mid, reason_code_list, properties):
    # Since we subscribed only for a single channel, reason_code_list contains
    # a single entry
    if reason_code_list[0].is_failure:
        print(f"Broker rejected you subscription: {reason_code_list[0]}")
    else:
        print(f"Broker granted the following QoS: {reason_code_list[0].value}")

def on_unsubscribe(client, userdata, mid, reason_code_list, properties):
    # Be careful, the reason_code_list is only present in MQTTv5.
    # In MQTTv3 it will always be empty
    if len(reason_code_list) == 0 or not reason_code_list[0].is_failure:
        print("unsubscribe succeeded (if SUBACK is received in MQTTv3 it success)")
    else:
        print(f"Broker replied with failure: {reason_code_list[0]}")
    client.disconnect()

def on_message(client, userdata, message):
    print(f"Message received on topic {message.topic}")
    # Decode the message payload
    decoded_content = base64.b64decode(message.payload)

    # Path to save the file
    output_file_path = "/home/houwei/Lab2/output.txt"

    # Write the decoded content to a file
    with open(output_file_path, "wb") as output_file:
        output_file.write(decoded_content)
    print(f"File has been written to {output_file_path}")

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code.is_failure:
        print(f"Failed to connect: {reason_code}. loop_forever() will retry connection")
    else:
        # we should always subscribe from on_connect callback to be sure
        # our subscribed is persisted across reconnections.
        topics = [("emotion/face", 2), ("emotion/voice", 2)]
        client.subscribe(topics)
        print("Subscribed to topics: emotion/face, emotion/voice")
		#for topic, qos in topics_to_subscribe:
		#	client.subscribe(topic, qos)
		#	print(f"Subscribed to {topic} with QoS {qos}")

mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqttc.on_connect = on_connect
mqttc.on_message = on_message
mqttc.on_subscribe = on_subscribe
mqttc.on_unsubscribe = on_unsubscribe

mqttc.user_data_set([])

# Using your pi as broker, connect to your pi's ip address; change the ip address below accordingly
mqttc.connect("192.168.26.142")
# Alternatively, I think can use the cloud as broker to test the code; uncomment the line below
# mqttc.connect("mqtt.eclipseprojects.io")

mqttc.loop_forever()
print(f"Received the following message: {mqttc.user_data_get()}")
