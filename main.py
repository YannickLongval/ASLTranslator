"""
Load in the cv and nlp models to read ASL characters from the webcam, and translate the signed words.
"""

import tensorflow as tf
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

"""Process the image to be passed into the model

Args:
    img (np.ndarray): the original image

Returns a 28x28, normalized respresentation of the image
"""
def preprocess(img:np.ndarray) -> np.ndarray:
    imgProcessed = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    print(imgProcessed/255)
    return imgProcessed/255

"""Passes the image into the model to predict what ASL character is shown

Args:
    model: the trained model to predict the ASL character
    img: the image of the ASL character

Returns the character that the image represents
"""
def predict(model, img: np.ndarray) -> str:
    # normalize img and put into np array
    normalized = np.array([preprocess(img)])
    predictions:list[str] = model.predict(normalized)
    return chr(65 + predictions[0].argmax())

# load the trained model from train.py
model = tf.keras.models.load_model('models/ASL_CV_model')

# use opencv to detect hands from the webcam
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# padding around the detected hands
offset = 20

# holds all of the letters the user signs
letters = []
while True:
    success, img = cap.read()
    hands, img_detected = detector.findHands(cv2.flip(img, 1), flipType=False)   
    if hands:
        # capture a grayscale, square image around the detected hand to be used for the model
        im_gray = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2GRAY)
        hand = hands[0]
        x, y, w, h = hand['bbox']
        if h > w:
            wOffset = (h - w)//2 + offset
            imgCrop = im_gray[y-offset:y+h+offset, x-wOffset:x+w+wOffset]
        else:
            hOffset = (w - h)//2 + offset
            imgCrop = im_gray[y-hOffset:y+h+hOffset, x-offset:x+w+offset]  
        predicted = predict(model, imgCrop)
    else:
         predicted = ""

    # fontScale
    fontScale = 3
    
    # Red color in BGR
    color = (0, 0, 255)
    
    # Line thickness of 2 px
    thickness = 4

    # Using cv2.putText() method

    img_detected = cv2.putText(img_detected, "".join(letters + [predicted]), (00, 185), cv2.FONT_HERSHEY_SIMPLEX, fontScale, 
                 color, thickness, cv2.LINE_AA, False)
        
    cv2.imshow("Image", img_detected)
    key = cv2.waitKey(1)

    # close program if esc key is pressed, save detecter character to letters if space bar is pressed
    if key%256 == 27:
            break
    if key%256 == 32 and predicted != "":
         letters.append(predicted)
         predicted = ""