import torch
import cv2
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
FRAMEWIDTH = 640
FRAMEHEIGHT = 480

# Initiate the Face Detection Cascade Classifier
HAARCASCADE = "haarcascade_frontalface_alt2.xml"
DEFAULT_DETECTOR = cv2.CascadeClassifier(HAARCASCADE)


#CV2 TEXT VARS
FONTSIZE = 1
FONTTHICKNESS = 1
TEXTCOLOR = (0,0,0)
FONTTYPE = cv2.FONT_HERSHEY_DUPLEX
LINE = cv2.LINE_AA

MARGIN = 10