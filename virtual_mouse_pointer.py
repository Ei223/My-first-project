import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import math

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

screen_w, screen_h = pyautogui.size()
smoothening = 8
plocX, plocY = 0, 0
clocX, clocY = 0, 0

pTime = 0
lastClick = 0
clickCooldown = 1  # seconds

# MediaPipe hand setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

def fingers_up(lm):
    fingers = []
    # Thumb
    fingers.append(1 if lm[4].x < lm[3].x else 0)
    # Other fingers
    fingers += [1 if lm[id].y < lm[id - 2].y else 0 for id in [8, 12, 16, 20]]
    return fingers

def get_landmark_pos(lm, shape, id):
    h, w, _ = shape
    return int(lm[id].x * w), int(lm[id].y * h)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb=np.ascontiguousarray(rgb,dtype=np.uint8)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark
        mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        fingers = fingers_up(lm)

        x1, y1 = get_landmark_pos(lm, frame.shape, 8)  # index tip
        screen_x = np.interp(x1, (0, 640), (0, screen_w))
        screen_y = np.interp(y1, (0, 480), (0, screen_h))

        # Smoothening + jitter filter
        if abs(screen_x - plocX) > 3 or abs(screen_y - plocY) > 3:
            clocX = plocX + (screen_x - plocX) / smoothening
            clocY = plocY + (screen_y - plocY) / smoothening
            pyautogui.moveTo(clocX, clocY)
            plocX, plocY = clocX, clocY

        # Left click: index + thumb close
        thumb_x, thumb_y = get_landmark_pos(lm, frame.shape, 4)
        if math.hypot(thumb_x - x1, thumb_y - y1) < 30 and time.time() - lastClick > clickCooldown:
            pyautogui.click()
            cv2.putText(frame, "Left Click", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            lastClick = time.time()

        # Right click: index + middle close
        mid_x, mid_y = get_landmark_pos(lm, frame.shape, 12)
        if fingers[1] and fingers[2] and math.hypot(x1 - mid_x, y1 - mid_y) < 30 and time.time() - lastClick > clickCooldown:
            pyautogui.rightClick()
            cv2.putText(frame, "Right Click", (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            lastClick = time.time()

        # Drag: all fingers down (fist)
        if sum(fingers) == 0:
            pyautogui.mouseDown()
            cv2.putText(frame, "Dragging...", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            pyautogui.mouseUp()

        # Scroll up (only middle finger up)
        if fingers[2] == 1 and sum(fingers) == 1:
            pyautogui.scroll(20)
            cv2.putText(frame, "Scroll Up", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Scroll down (only ring finger up)
        if fingers[3] == 1 and sum(fingers) == 1:
            pyautogui.scroll(-20)
            cv2.putText(frame, "Scroll Down", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime + 1e-6)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2)

    cv2.imshow("Virtual Mouse (Advanced)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
