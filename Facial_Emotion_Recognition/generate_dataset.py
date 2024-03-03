from torch.utils.data import Dataset
import os
import pandas as pd
from modules.FERDataset import EmotionRecognitionDataset
        
# Generates train,val and test datasets for reproducibility
def generate_datasets(dataframe,data_path):
    if(not(os.path.exists(f'{data_path}/training.csv'))):
        train_df = dataframe[dataframe['Usage'] == 'Training'] 
        val_df = dataframe[dataframe['Usage'] == 'PublicTest'] 
        test_df = dataframe[dataframe['Usage'] == 'PrivateTest'] 

        train_df.to_csv(f'{data_path}/training.csv')
        val_df.to_csv(f'{data_path}/val.csv')
        test_df.to_csv(f'{data_path}/test.csv')
        
def main(): 
    main_dataset_path = "FER2013"
    main_dataset = EmotionRecognitionDataset(main_dataset_path,None)
    save_path = "datasets"
    generate_datasets(main_dataset.df,save_path)
    
if __name__ == "__main__":
    main()
    