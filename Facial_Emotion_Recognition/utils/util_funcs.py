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


def get_train_transform():
    transform_1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(0.5),
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