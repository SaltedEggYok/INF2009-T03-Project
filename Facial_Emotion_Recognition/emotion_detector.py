import torch
import os
from modules.FERDataset import EmotionRecognitionDataset
from utils.config import DEVICE
import torch.nn as nn
from torch.utils.data import DataLoader
from model.FERModel import EmotionRecognitionModel
from mini_XCeption.XCeptionModel import Mini_Xception
from utils.util_funcs import get_train_transform,get_val_transform,get_generalized_emotion_map
import torchvision.transforms as transforms
import cv2

def convert_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image,(48,48))
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image = transforms.ToTensor()(image).to(DEVICE)
    image = torch.unsqueeze(image,0)
    return image

def main():
    base_model = Mini_Xception() 
    base_model.to(DEVICE)
    best_model = EmotionRecognitionModel(model = base_model,device = DEVICE,weights="ERM_Results/ERModel.pt")
    converted_image_happy = convert_image('test_happy.jpg')
    converted_image_sad = convert_image('test_sad.jpg')
    converted_image_neutral = convert_image('neutral_specific.jpg')
    result1 = best_model.predict_one(converted_image_happy)
    result2 = best_model.predict_one(converted_image_sad)
    result3 = best_model.predict_one(converted_image_neutral)
    print(get_generalized_emotion_map(result1))
    print(get_generalized_emotion_map(result2))
    print(get_generalized_emotion_map(result3))
   
    
if __name__ == "__main__":
    main()