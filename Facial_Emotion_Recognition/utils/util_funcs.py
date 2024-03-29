import torchvision.transforms as transforms
# Retrieves string form of emotion given predicted label
def get_emotion(label):
    label_emotion_map = { 
        0: 'Angry',
        1: 'Disgust', 
        2: 'Fear', 
        3: 'Happy', 
        4: 'Sad', 
        5: 'Surprise', 
        6: 'Neutral'        
    }
    return label_emotion_map[label]

def get_average_emotion(time_stampdict : dict):
    average_emotions = {}
    for time_stamp,emotion_dict in time_stampdict.items():
        hour, minute, second, _ = map(int, time_stamp.split(':'))
        average_emotions[(hour,minute,second)] = {}
        for person,emotions in emotion_dict.items():
            total_emotions = len(emotions)
            neutral_count = emotions.count('Neutral')
            positive_count = emotions.count('Positive')
            negative_count = emotions.count('Negative')
            
            if neutral_count >= positive_count and neutral_count >= negative_count:
                average_emotion = 'Neutral'
            elif positive_count >= neutral_count and positive_count >= negative_count:
                average_emotion = 'Positive'
            else:
                average_emotion = 'Negative'
            average_emotions[(hour,minute,second)][person] = average_emotion
    return average_emotions            

def get_generalized_emotion(emotion):
    if(emotion in [0,1,2,4,5]):
        return 0
    elif(emotion in [3]):
        return 1
    else:
        return 2

def get_generalized_emotion_map(emotion):
    label_emotion_map = \
    { 
        0: 'Negative',
        1: 'Positive', 
        2: 'Neutral'     
    }
    return label_emotion_map[emotion]
    
def get_train_transform():
    transform_1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(0.2),
            transforms.ToTensor()
        ]
    )
    return transform_1

def get_val_transform():
    val_transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    return val_transform