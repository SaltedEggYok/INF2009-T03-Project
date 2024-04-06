# import SER.source.SER_model as SER_model
# import SER.source.SER_trainmodel as SER_trainmodel
import SER_model
import SER_trainmodel
import numpy as np
import pyaudio
import wave
from datetime import datetime
from scipy.io.wavfile import write
from array import array
import cv2

from MQTT.mqtt_publisher_class import MQTTPublisher
# import MQTT.mqtt_publisher_class as publisher

if __name__ == "__main__":
    # train model
    # ser_train = SER_trainmodel.SER_trainmodel()
    # ser_train.add_audio_features("SER/datasets/our_data/")

    # run model
    ser_model = SER_model.SER_model()
    ser_model.load_model(
        "SER/saved_models/model.json", "SER/saved_models/Emotion_Voice_Detection_Model.h5")
    # ser_model.preprocess_audio("SER/datasets/our_data/")
    # ser_model.model_predict("SER/datasets/our_data/(put file here)")

    # audio parameters
    # samples per frame (you can change the same to acquire more or less samples)
    BUFFER = 1024 * 16
    FORMAT = pyaudio.paInt16    # audio format (bytes per sample)
    CHANNELS = 1                # single channel for microphone
    RATE = 44100                # samples per second
    RECORD_SECONDS = 4         # Specify the time to record from the microphone in seconds

    # create pyaudio instantiation
    audio = pyaudio.PyAudio()

    # Initialize a non-silent signals array to state "True" in the first 'while' iteration.
    data = array('h', np.random.randint(size=BUFFER, low=0, high=500))

    # perform audio recording every few seconds
    while (True):  # TODO : replace this boolean with a controllable one
        try:
            # create pyaudio stream
            stream = audio.open(format=FORMAT, channels=CHANNELS,
                                rate=RATE, input=True,
                                frames_per_buffer=BUFFER)

            # reset variables that are used to read buffer and store data
            frames = []
            data = np.nan

            print("start recording")  # DEBUG
            # get timestamp of when recording starts
            timestamp = datetime.now().strftime("%H:%M:%S:%f")
            split_time = timestamp.split(':')
            # get the milliseconds from microseconds
            milliseconds = int(split_time[3][0:3])
            # format the time properly
            formatted_time = f"{split_time[0]}:{split_time[1]}:{split_time[2]}:{str(milliseconds).zfill(3)}"

            # because files cannot use : in their names
            file_timestamp = formatted_time.replace(":", "_")

            # calculate number of timesteps
            timesteps = int(RATE / BUFFER * RECORD_SECONDS)

            # record audio for RECORD_SECONDS
            for i in range(0, timesteps):
                # read audio data
                data = stream.read(BUFFER)
                # append data
                frames.append(data)
                # print(np_data)

            # stream.stop_stream()
            # stream.close()
            # audio.terminate()

            print("recording complete")  # DEBUG

            # write audio to file
            wf = wave.open(
                f"./SER/our_data/testing/{file_timestamp}.wav", 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            print("audio saved " + file_timestamp)  # DEBUG

            # predict emotion
            pred = ser_model.model_predict(
                f"./SER/our_data/testing/{file_timestamp}.wav", timestamp=formatted_time)

            # escape loop
            k = cv2.waitKey(1) & 0xFF
            # press 'q' to exit
            if k == ord('q'):
                break
            # export all preds to excel
            ser_model.export_result(f"SER/pred/outputs_{file_timestamp}.txt")
        except KeyboardInterrupt:
            break

    # open the final 
    # mqtt_publisher = MQTTPublisher()
    #mqtt_publisher.connect()
    #mqtt_publisher.publish_payload("emotion/face", f"SER/pred/outputs_{file_timestamp}.txt")
    ser_model.sync_time()




