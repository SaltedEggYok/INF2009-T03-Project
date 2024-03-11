import torch
import numpy as np
from modules.FERDataset import EmotionRecognitionDataset
from utils.config import DEVICE
from utils.config import FRAMEHEIGHT,FRAMEWIDTH,DEFAULT_DETECTOR,FONTTHICKNESS,FONTSIZE,TEXTCOLOR,MARGIN,FONTTYPE,LINE
from torch.utils.data import DataLoader
from model.FERModel import EmotionRecognitionModel
from mini_XCeption.XCeptionModel import Mini_Xception
from utils.util_funcs import get_generalized_emotion_map
import torchvision.transforms as transforms
import cv2
import time

def write_image(image_path,image):
    # image = cv2.cvtColor(image.astype(np.float32),cv2.COLOR_GRAY2RGB)
    cv2.imwrite(image_path,image)
    

def convert_image(image):
    # image = cv2.imread(image_path)
    # image = cv2.resize(np.uint8(image),(48,48),interpolation=cv2.INTER_AREA)
    image = cv2.resize(np.uint8(image),(48,48))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    write_image('face.jpg',image=image)
    print(image.shape)
    image = transforms.ToTensor()(image).to(DEVICE)
    # print(image.shape)
    image = torch.unsqueeze(image,0)
    print(image.shape)
    return image

def convert_grayscale(image):
     return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

def main():
    base_model = Mini_Xception() 
    base_model.to(DEVICE)
    best_model = EmotionRecognitionModel(model = base_model,device = DEVICE,weights="ERM_Results/ERModel.pt")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAMEWIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAMEHEIGHT)
    while True:
        try:
            time.sleep(0.05)
            ret, frame = cap.read()
            # frame = cv2.flip(frame, 1) # To flip the image to match with camera flip
            grayscale_image = convert_grayscale(frame)
            faces = DEFAULT_DETECTOR.detectMultiScale(grayscale_image)
            # Get emotion of each face detected
            for face in faces:
                (x,y,w,d) = face
                face = frame[y:y+d,x:x+w]
                face = convert_image(face)
                pred = best_model.predict_one(face)
                result_text = get_generalized_emotion_map(pred)
                text_location = (MARGIN + x,
                                MARGIN + y)
                cv2.rectangle(frame,(x,y),(x+w, y+d),(255, 255, 255), 2)
                cv2.putText(frame, result_text, text_location, FONTTYPE,
                            FONTSIZE, TEXTCOLOR, FONTTHICKNESS, LINE)
            cv2.imshow('Feed',frame)
            key = cv2.waitKey(1)
            if(key == ord('q')):
                break
        except KeyboardInterrupt:
            break
    cap.release()
    cv2.destroyAllWindows()
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