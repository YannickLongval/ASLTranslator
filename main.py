"""
Load in the cv and nlp models to read ASL characters from the webcam, and translate the signed words.
"""

import tensorflow as tf
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

def preprocess(img):
    imgProcessed = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    print(imgProcessed/255)
    return imgProcessed/255

def predict(model, img):
    # normalize img and put into np array
    normalized = np.array([preprocess(img)])
    predictions:list[str] = model.predict(normalized)
    return chr(65 + predictions[0].argmax())

# load the trained model from train.py
model = tf.keras.models.load_model('models/ASL_CV_model')

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
while True:
    success, img = cap.read()
    hands, img_detected = detector.findHands(cv2.flip(img, 1), flipType=False)   
    if hands:
        im_gray = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2GRAY)
        hand = hands[0]
        x, y, w, h = hand['bbox']
        if h > w:
            wOffset = (h - w)//2 + offset
            imgCrop = im_gray[y-offset:y+h+offset, x-wOffset:x+w+wOffset]
        else:
            hOffset = (w - h)//2 + offset
            imgCrop = im_gray[y-hOffset:y+h+hOffset, x-offset:x+w+offset]  
        
        # fontScale
        fontScale = 1
        
        # Red color in BGR
        color = (0, 0, 255)
        
        # Line thickness of 2 px
        thickness = 2
   
        # Using cv2.putText() method
        img_detected = cv2.putText(img_detected, predict(model, imgCrop), (00, 185), cv2.FONT_HERSHEY_SIMPLEX, fontScale, 
                 color, thickness, cv2.LINE_AA, False)
        
    cv2.imshow("Image", img_detected)
    key = cv2.waitKey(1)
    if key%256 == 27:
            break