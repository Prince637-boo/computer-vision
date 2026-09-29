import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


while True: 
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    cv2.imshow("Image", img)
    cv2.waitKey(1)