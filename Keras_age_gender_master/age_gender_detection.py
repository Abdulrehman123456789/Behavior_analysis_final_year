
import cv2
import os
from time import sleep
import numpy as np
import argparse
from Keras_age_gender_master.wide_resnet import WideResNet
from keras.utils.data_utils import get_file

faceCVobject=None

class FaceCV(object):
    
    CASE_PATH = "pretrained_models/haarcascade_frontalface_alt.xml"
    WRN_WEIGHTS_PATH = "https://github.com/Tony607/Keras_age_gender/releases/download/V1.0/weights.18-4.06.hdf5"


    def __new__(cls, weight_file=None, depth=16, width=8, face_size=64):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceCV, cls).__new__(cls)
        return cls.instance

    def __init__(self, depth=16, width=8, face_size=64):
        self.face_size = face_size
        self.model = WideResNet(face_size, depth=depth, k=width)()
        model_dir = os.path.join(os.getcwd(), "pretrained_models").replace("//", "\\")
        fpath = get_file('weights.18-4.06.hdf5',
                         self.WRN_WEIGHTS_PATH,
                         cache_subdir=model_dir)
        self.model.load_weights(fpath)

    @classmethod
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    def crop_face(self, imgarray, section, margin=40, size=64):
       
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w,h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w-1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h-1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

    def detect_face(self):
        finalstr=""

        frame= cv2.imread("face.png")
        h,w,d=frame.shape
        face=[0,0,w,h]
        face_img,cropped = self.crop_face(frame, face, margin=40, size=self.face_size)
        (x, y, w, h) = cropped
        face_imgs = np.empty((1, self.face_size, self.face_size, 3))
        face_imgs[0,:,:,:] = face_img
        results = self.model.predict(face_imgs)
        predicted_genders = results[0]
        if predicted_genders[0][0]>0.5:
            finalstr=finalstr+"F"
            print("F")
        else:
            finalstr=finalstr+"M"
            print("M")
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()
        print(int(predicted_ages[0]))
        finalstr=finalstr+"/"+str(int(predicted_ages[0]))
        return(finalstr)

def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    args = parser.parse_args()
    return args

def initialize_age_gender_detection_model_loading_process():

    global faceCVobject

    args = get_args()
    depth = args.depth
    width = args.width

    faceCVobject = FaceCV(depth=depth, width=width)

    


def age_gender_detect():

    global faceCVobject
    finalvalue=faceCVobject.detect_face()
    return(finalvalue)

