import os
import cv2
import pandas as pd
import numpy as np

emotion_mapping = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6
}

# Function to preprocess the image
def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Resize to 48 by 48
    image = cv2.resize(np.uint8(image),(48,48),interpolation=cv2.INTER_AREA)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    return gray_image

# Function to convert image data to CSV format
def convert_to_csv(image_number, emotion_label, image_data, usage):
    # Flatten the pixel values
    flattened_pixels = image_data.flatten()
    # Convert pixel values to comma-separated string
    pixel_string = ' '.join(map(str, flattened_pixels))
    # Construct CSV row
    csv_row = f"{image_number},{emotion_label},{pixel_string},{usage}"
    return csv_row

# Path to the directory containing image data
input_path = 'datasets/our_data/'

# Initialize an empty list to store CSV rows
csv_data = []

# Determine the last image number in training.csv if it exists
if os.path.exists('datasets/training.csv'):
    training_df = pd.read_csv('datasets/training.csv')
    image_number = training_df.iloc[-1,0] + 1

# Loop through each emotion label folder e.g. angry, disgust, fear,... 
for emotion_label_name in os.listdir(input_path):
    # Convert emotion label mapped to numbers 0-6
    emotion_label = emotion_mapping[emotion_label_name]
    # Retrieve emotion folder path
    emotion_folder_path = os.path.join(input_path, emotion_label_name)
    if os.path.isdir(emotion_folder_path):
        # Loop through each image file in the emotion label folder
        for image_file in os.listdir(emotion_folder_path):
            # Retrieve image file path
            image_path = os.path.join(emotion_folder_path, image_file)
            # Determine usage based on the image file name
            # if "Training" in image_file:
            usage = "Training"
                
            # # If needed for Test folder
                
            # # elif "PrivateTest" in image_file:
            # #     usage = "PrivateTest"
            # # elif "PublicTest" in image_file:
            # #     usage = "PublicTest"
            # else:
            #     raise ValueError("Unknown usage type in file name")
            
            # Preprocess the image
            pixels = preprocess_image(image_path)
            # Convert image data to CSV format
            csv_row = convert_to_csv(image_number, emotion_label, pixels, usage)
            # Append CSV row to the list
            csv_data.append(csv_row)
            image_number += 1

# Append CSV data to training.csv
with open('datasets/training.csv', 'a') as file:
    for row in csv_data:
        file.write('\n' + row)

print("Data appended to training.csv successfully.")

# # Write CSV data to file
# with open('output.csv', 'w') as file:
#     file.write(",emotion,pixels,Usage\n")  # Write CSV header
#     for row in csv_data:
#         file.write(row + '\n')  # Write CSV rows

# print("CSV file created successfully.")