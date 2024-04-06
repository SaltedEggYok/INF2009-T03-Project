import torch
import numpy as np
from utils.config import DEVICE
from utils.config import FRAMEHEIGHT,FRAMEWIDTH,DEFAULT_DETECTOR,FONTTHICKNESS,FONTSIZE,TEXTCOLOR,MARGIN,FONTTYPE,LINE
from model.FERModel import EmotionRecognitionModel
from mini_XCeption.XCeptionModel import Mini_Xception
from utils.util_funcs import get_generalized_emotion_map,get_average_emotion
from datetime import datetime
import torchvision.transforms as transforms
import cv2
import time
import pandas as pd
from MQTT.mqtt_publisher_class import MQTTPublisher

def write_image(image_path,image):
    # image = cv2.cvtColor(image.astype(np.float32),cv2.COLOR_GRAY2RGB)
    cv2.imwrite(image_path,image)
    
def convert_image(image):
    # image = cv2.imread(image_path)
    # image = cv2.resize(np.uint8(image),(48,48),interpolation=cv2.INTER_AREA)
    image = cv2.resize(np.uint8(image),(48,48),interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    write_image('face.jpg',image=image)
    image = transforms.ToTensor()(image).to(DEVICE)
    # print(image.shape)
    image = torch.unsqueeze(image,0)
    return image

def convert_grayscale(image):
     return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

def sync_time(results,start_time,output_file_name):
    # open the final output file
    output_file = open(output_file_name, "w")

    starting_hour,starting_minute,starting_second = map(int, start_time.split(':'))
    # access dictionary
    if len(results) > 0:
        # iterate through the dictionary
        for key, value in results.items():
            # if it's the first element, set the starting time to be used to offset all other times      
            # offset all times by the starting time
            hour, minute, second = key
            # if the hour is less than the starting hour, it means it's the next day
            if hour < starting_hour:
                hour += 24  
            # if the minute is less than the starting minute, it means the hour is less than the starting hour, same for seconds
            if minute < starting_minute:
                hour -= 1
                minute += 60
            if second < starting_second:
                minute -= 1
                second += 60

            # write the offset time to the file
            output_key = (hour - starting_hour, minute - starting_minute, second - starting_second)
            output_file.write(f"{output_key}: {value}\n")
def main():
    base_model = Mini_Xception() 
    base_model.to(DEVICE)
    best_model = EmotionRecognitionModel(model = base_model,device = DEVICE,weights="ERM_Results/ERModel.pt")
    cap = cv2.VideoCapture(0)
    frame_list = []
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAMEWIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAMEHEIGHT)
    time_stamp_dict = {}
    start_time = datetime.now().strftime("%H:%M:%S")
    while True:
        try:
            ret, frame = cap.read()
            # frame = cv2.flip(frame, 1) # To flip the image to match with camera flip
            grayscale_image = convert_grayscale(frame)
            faces = DEFAULT_DETECTOR.detectMultiScale(grayscale_image)
            # Get emotion of each face detected
            face_dict = {}
            time_stamp = datetime.now().strftime("%H:%M:%S:%f")
            split_time = time_stamp.split(':')
            # get the milliseconds from microseconds
            milliseconds = int(split_time[3][0:3])
            # format the time properly
            formatted_time = f"{split_time[0]}:{split_time[1]}:{split_time[2]}:{str(milliseconds).zfill(3)}"
            for idx,face in enumerate(faces):
                # time_stamp = datetime.now()
                (x,y,w,d) = face
                # get position of face relative to frame
                face = frame[y:y+d,x:x+w]
                # pre process image
                face = convert_image(face)
                pred = best_model.predict_one(face)
                result_text = get_generalized_emotion_map(pred)

                # add person:emotion dictionary to time_stamp_dict
                face_dict[idx] = [result_text]
                time_stamp_dict[formatted_time] = face_dict
                
                text_location = (MARGIN + x,
                                MARGIN + y)
                cv2.rectangle(frame,(x,y),(x+w, y+d),(255, 255, 255), 2)
                cv2.putText(frame, result_text, text_location, FONTTYPE,
                            FONTSIZE, TEXTCOLOR, FONTTHICKNESS, LINE)
                
            cv2.imshow('Feed',frame)
            frame_list.append(frame)
            key = cv2.waitKey(1)
            #out.write(frame)
            if(key == ord('q')):
                break
        except KeyboardInterrupt:
            break            
    cap.release()
    cv2.destroyAllWindows()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('video/video.mp4',fourcc,1000/60.0,(640,480))
    for i in range(len(frame_list)):
        out.write(frame_list[i])
    emotion_dict = get_average_emotion(time_stamp_dict)
    sync_time(emotion_dict,start_time,'ERM_Results/emotion_dict.txt')
    
    mqtt_publisher = MQTTPublisher()
    mqtt_publisher.connect()
    mqtt_publisher.publish_payload("emotion/facial",'ERM_Results/emotion_dict.txt')
    mqtt_publisher.disconnect()
    
    
    # with open('ERM_Results/emotion_dict.txt', 'w') as file:
    #     for key, value in emotion_dict.items():
    #         file.write(f"{key}: {value}\n")
    
    
    # converted_image_happy = convert_image('test_happy.jpg')
    # converted_image_sad = convert_image('test_sad2.jpg')
    # converted_image_neutral = convert_image('test_angry.jpg')
    # result1 = best_model.predict_one(converted_image_happy)
    # result2 = best_model.predict_one(converted_image_sad)
    # result3 = best_model.predict_one(converted_image_neutral)
    # print(get_generalized_emotion_map(result1))
    # print(get_generalized_emotion_map(result2))
    # print(get_generalized_emotion_map(result3))
if __name__ == "__main__":
    main()