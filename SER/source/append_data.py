import os
import re
import shutil


def validate_filename(filename):

    '''
    Validates the format of the filenames of audio files in a folder
    :param filename: Path to folder of audio files
    :type filename: str
    :return: True if ALL filenames are valid, False otherwise
    '''
    
    # Define regular expression pattern for filename validation
    pattern = r'^\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}\.(wav|mp4)$'
    
    # Validate filename format
    if not re.match(pattern, filename):
        print(f"Invalid filename format: {filename}")
        return False
    
    # Extract parts of the filename
    parts = filename.split('-')
    
    modality = parts[0]
    channel = parts[1]
    emotion = parts[2]
    intensity = parts[3]
    statement = parts[4]
    repetition = parts[5]
    gender = parts[6].split('.')[0]  # Remove file extension

    # print("Modality:", modality)
    # print("Channel:", channel)
    # print("Emotion:", emotion)
    # print("Intensity:", intensity)
    # print("Statement:", statement)
    # print("Repetition:", repetition)
    # print("Gender:", gender)
    
    # Validate modality
    if modality not in {'01', '02', '03'}:
        print(f"\nInvalid modality in filename: {filename}")
        return False
    
    # Validate emotion
    if emotion not in {'01', '02', '03', '04', '05', '06', '07', '08'}:
        print(f"\nInvalid emotion in filename: {filename}")
        return False
    
    # If all validations passed
    return True

def process_files(input_directory):
    '''
    Processes audio files in a directory and copies them to appropriate destination directories
    :param input_directory: Path to folder of audio files
    :type input_directory: str
    :return: None
    '''
    
    # Define destination directories
    base_dest_directory = os.path.join(dir, 'data', 'training')
        
    # Create destination directory if they don't exist
    os.makedirs(base_dest_directory, exist_ok=True)
    
    # Get list of files in the input directory
    files = os.listdir(input_directory)
    
    # Process each file
    for file in files:
        if validate_filename(file):
            # print(f"Valid filename: {file}")
            # Extract gender from filename
            gender = int(file.split('-')[6].split('.')[0])
            dest_directory = os.path.join(base_dest_directory, f'Actor_{gender:02}')
  
            # Create actor directory if it doesn't exist
            os.makedirs(dest_directory, exist_ok=True)

            dest_file_path = os.path.join(dest_directory, file)

            # Check if file already exists in destination directory
            if os.path.exists(dest_file_path):
                print(f"File {file} already exists in {dest_directory}")
            else:
                # Copy file to appropriate directory
                try:
                    shutil.copy(os.path.join(input_directory, file), dest_file_path)
                    print(f"File copied to {dest_directory}")
                except Exception as e:
                    print(f"Error copying file: {e}")
        else:
            print(f"Skipping invalid filename: {file}\n")

# Example usage
dir = "SER/"
process_files(dir + "our_data")
