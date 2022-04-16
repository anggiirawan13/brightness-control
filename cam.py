import cv2
import mediapipe as mp
import screen_brightness_control as sbc
import numpy as np
from math import hypot

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    lmList = []
    
    if results.multi_hand_landmarks:
        for handlandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):
                h, v, _ = img.shape
                cx, cy = int(lm.x*v), int(lm.y*h)
                lmList.append([id, cx, cy])
            mp_draw.draw_landmarks(img, handlandmark, mp_hands.HAND_CONNECTIONS)

    if lmList != []:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        cv2.circle(img, (x1, y1), 4, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 4, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2),(255,0, 0), 3)

        length = hypot(x2-x1, y2-y1)
        bright = np.interp(length, [50,300], [0,100])

        sbc.set_brightness(int(bright))

    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break