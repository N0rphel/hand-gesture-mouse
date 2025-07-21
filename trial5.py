import cv2 as cv
import mediapipe as mp
import pyautogui
import time
import math

# Screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
last_click_time = 0
click_delay = 0.3  # seconds

# Initialize camera
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Function to compute Euclidean distance
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Function to get finger states
def get_finger_states(hand_landmarks, hand_label):
    finger_states = []
    lm = hand_landmarks.landmark

    # Thumb: x comparison
    if hand_label == "Right":
        finger_states.append(1 if lm[4].x < lm[3].x else 0)
    else:
        finger_states.append(1 if lm[4].x > lm[3].x else 0)

    # Other fingers: y comparison (tip.y < pip.y → extended)
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        finger_states.append(1 if lm[tip].y < lm[pip].y else 0)

    return finger_states

# Smoothing setup
smoothening = 7
plocX, plocY = 0, 0

# Scrolling setup
prev_scroll_y = None
scroll_threshold = 15  # Pixels to trigger scroll

# Main loop
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv.flip(frame, 1)  # Mirror image
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)

    if output.multi_hand_landmarks and output.multi_handedness:
        for hand_landmarks, handedness in zip(output.multi_hand_landmarks, output.multi_handedness):
            hand_label = handedness.classification[0].label

            if hand_label == 'Right':
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                finger_states = get_finger_states(hand_landmarks, hand_label)
               # print("Finger states:", finger_states)

                # Scroll when palm is open
                if finger_states == [1, 1, 1, 1, 1]:
                    middle_finger_mcp = hand_landmarks.landmark[9].y * frame_height

                    if prev_scroll_y is not None:
                        dy = middle_finger_mcp - prev_scroll_y
                        if abs(dy) > scroll_threshold:
                            #print(dy*0.05)
                            pyautogui.scroll(-int(dy)*0.05)  # Scroll up/down
                    prev_scroll_y = middle_finger_mcp
                    continue  # Skip other actions when scrolling
                else:
                    prev_scroll_y = None  # Reset scroll tracker

                # Move cursor only if index is up and middle is down
                if finger_states[1] == 1 and finger_states[2] == 0:
                    index_tip = hand_landmarks.landmark[8]
                    x = int(index_tip.x * SCREEN_WIDTH)
                    y = int(index_tip.y * SCREEN_HEIGHT)

                    # Apply smoothing
                    clocX = plocX + (x - plocX) / smoothening
                    clocY = plocY + (y - plocY) / smoothening
                    pyautogui.moveTo(clocX, clocY)
                    plocX, plocY = clocX, clocY

                # Perform click if index and middle are both up and close together
                if finger_states[1] == 1 and finger_states[2] == 1:
                    index_tip = hand_landmarks.landmark[8]
                    middle_tip = hand_landmarks.landmark[12]

                    # Convert to pixel coordinates
                    index_px = (int(index_tip.x * frame_width), int(index_tip.y * frame_height))
                    middle_px = (int(middle_tip.x * frame_width), int(middle_tip.y * frame_height))

                    dist = euclidean_distance(index_px, middle_px)

                    if dist < 30 and (time.time() - last_click_time > click_delay):
                        pyautogui.click()
                        last_click_time = time.time()

    cv.imshow('AI Mouse Control', frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
