from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import cv2
# Custom Dataset Class
class EmotionRecognitionDataset(Dataset):
    def __init__(self,root,transforms):
        self.root = root
        self.transform = transforms
        # If the datasets exist, use the respective dataset, otherwise, use the main root dataset 
        if(not(os.path.exists('datasets/training.csv'))):
            self.csv_path = os.path.join(self.root,'fer20131.csv')
        else:
            self.csv_path = self.root
        self.df = pd.read_csv(self.csv_path)
    
    def __len__(self):
        return self.df.index.size
    
    def __getitem__(self, index):
        df_row = self.df.iloc[index]
        row_emotion = df_row['emotion']
        row_pixels = df_row['pixels']
        # convert to np array
        row_face = list(map(int,row_pixels.split(' ')))
        # reshape to 48 by 48 as dataset is 48 x 48
        row_face = np.array(row_face).reshape(48,48).astype(np.uint8)
        if self.transform is not None:
            row_face = cv2.equalizeHist(row_face)
            row_face = self.transform(row_face)
        return row_face,row_emotion


