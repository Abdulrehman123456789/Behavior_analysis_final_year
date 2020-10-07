from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

# parameters for loading data and images
emotion_model_path = 'Emotion_recognition_master/models/_mini_XCEPTION.102-0.66.hdf5'

# hyper-parameters for bounding boxes shape
# loading models

emotion_classifier=None

def initialize_emotion_detection_model_loading_process():
    global emotion_classifier
    emotion_classifier = load_model(emotion_model_path, compile=False)
    
def emotion_detect():
    global emotion_classifier
    frame=cv2.imread("face.png")
    #reading the frame
    frame = imutils.resize(frame,width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray
    roi = cv2.resize(roi, (64, 64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
        
        
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]
    label = EMOTIONS[preds.argmax()]
    print(label)
    return(label)
    