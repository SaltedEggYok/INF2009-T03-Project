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