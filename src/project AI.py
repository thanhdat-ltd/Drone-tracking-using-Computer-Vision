import cv2
import time
import mediapipe as mp
import numpy as np
import threading
import pyautogui
import os
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# --- BUFFER TRUNG BÌNH / EMA ---
buffer_size = 7
pos_x_buffer = deque(maxlen=buffer_size)
pos_y_buffer = deque(maxlen=buffer_size)

# --- CẤU HÌNH LỌC MƯỢT ---
filter_mode = "ema"  # Chọn: "mean", "ema", "none"
ema_alpha = 0.2       # Hệ số cho EMA
ema_x = None
ema_y = None

# --- GLOBAL CONFIGURATION ---
global_click_lock = threading.Lock()
global_click_frame = None

prev_pixel_x = None
prev_pixel_y = None
smoothed_x, smoothed_y = pyautogui.position()

speed = 7.0
sensitivity = 1
click_cooldown = 0.5
last_click_time = 0

# Thời gian FPS
delta_threshold = 2
target_fps = 30

# --- ILoveYou detection ---
iloveyou_start_time = None
no_iloveyou_start_time = None
exit_flag = False

# --- CẤU HÌNH MÔ HÌNH GESTURE ---
base_options_gesture = python.BaseOptions(model_asset_path='C:/Users/datso/Downloads/Compressed/src/models/gesture_recognizer.task')
gesture_options = vision.GestureRecognizerOptions(
    base_options=base_options_gesture,
    num_hands=2
)
gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_options)

# --- Công cụ vẽ ---
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = gesture_recognizer.recognize(mp_image)
    return result.gestures, result.handedness, result.hand_landmarks

def get_index_finger_position(hand_landmarks, image_shape):
    if not hand_landmarks:
        return None, None
    h, w, _ = image_shape
    if hasattr(hand_landmarks, 'landmark'):
        landmarks = hand_landmarks.landmark
    else:
        landmarks = hand_landmarks
    index_tip = landmarks[8]
    pixel_x = int((1 - index_tip.x) * w)
    pixel_y = int(index_tip.y * h)
    return pixel_x, pixel_y

def lerp(a, b, f):
    return a + f * (b - a)

def click_processing():
    global global_click_frame, prev_pixel_x, prev_pixel_y, last_click_time
    global smoothed_x, smoothed_y, ema_x, ema_y
    global iloveyou_start_time, no_iloveyou_start_time, exit_flag
    open_palm_start_time = None

    while True:
        start_time = time.time()

        # Thoát nếu đã flag
        if exit_flag:
            break

        frame_to_process = None
        with global_click_lock:
            frame_to_process = global_click_frame.copy() if global_click_frame is not None else None
        if frame_to_process is None:
            continue

        gestures_result, handedness_result, hand_landmarks_result = process_frame(frame_to_process)
        if not hand_landmarks_result:
            prev_pixel_x, prev_pixel_y = None, None
            continue

        # --- KIỂM TRA ILoveYou 2s với reset sau 0.5s break ---
        detected_iloveyou = any(
            gestures_result[i] and gestures_result[i][0].category_name.lower() == "iloveyou"
            for i in range(len(gestures_result))
        )
        now = time.time()
        if detected_iloveyou:
            # Reset absence timer
            no_iloveyou_start_time = None
            if iloveyou_start_time is None:
                iloveyou_start_time = now
            elif now - iloveyou_start_time >= 3.0:
                print("[DEBUG] Detected 'iloveyou' ≥ 2s, exiting...")
                os._exit(0)
        else:
            # Start absence timer if not already
            if no_iloveyou_start_time is None:
                no_iloveyou_start_time = now
            elif now - no_iloveyou_start_time >= 0.5:
                # Reset main timer after 0.5s without gesture
                iloveyou_start_time = None
                no_iloveyou_start_time = None

        height, width, _ = frame_to_process.shape
        screen_width, screen_height = pyautogui.size()
        right_hand_found = False
        mouse_locked = False

        # Khóa chuột nếu tay trái Victory
        for i in range(len(hand_landmarks_result)):
            label = handedness_result[i][0].category_name
            gesture = gestures_result[i][0].category_name.lower() if gestures_result[i] else None
            if label == "Left" and gesture == "victory":
                mouse_locked = True
                break

        # Xử lý từng tay
        for i in range(len(hand_landmarks_result)):
            label = handedness_result[i][0].category_name
            gesture = gestures_result[i][0].category_name.lower() if gestures_result[i] else None

            # Tay trái điều khiển click/scroll
            if label == "Left":
                if gesture == "open_palm":
                    if open_palm_start_time is None:
                        open_palm_start_time = now
                    elif now - open_palm_start_time >= 3:
                        pyautogui.moveTo(screen_width//2, screen_height//2)
                        open_palm_start_time = None
                else:
                    open_palm_start_time = None

                if gesture == "closed_fist":
                    if time.time() - last_click_time > click_cooldown:
                        pyautogui.click()
                        last_click_time = time.time()
                elif gesture == "thumb_up":
                    pyautogui.scroll(100)
                elif gesture == "thumb_down":
                    pyautogui.scroll(-100)

            # Tay phải điều khiển di chuyển chuột
            if label == "Right" and not mouse_locked:
                right_hand_found = True
                px, py = get_index_finger_position(hand_landmarks_result[i], frame_to_process.shape)
                if px is None:
                    continue
                if prev_pixel_x is None:
                    prev_pixel_x, prev_pixel_y = px, py
                    curr_x, curr_y = pyautogui.position()
                    smoothed_x, smoothed_y = curr_x, curr_y
                    ema_x, ema_y = curr_x, curr_y
                else:
                    dt = max(1/target_fps, time.time() - start_time)
                    dx = (px - prev_pixel_x) * speed * sensitivity * dt * 50
                    dy = (py - prev_pixel_y) * speed * sensitivity * dt * 50
                    curr_x, curr_y = pyautogui.position()
                    new_x = np.clip(curr_x + dx, 0, screen_width)
                    new_y = np.clip(curr_y + dy, 0, screen_height)
                    ema_x = ema_alpha * new_x + (1-ema_alpha) * ema_x
                    ema_y = ema_alpha * new_y + (1-ema_alpha) * ema_y
                    smoothed_x = lerp(curr_x, ema_x, 0.3)
                    smoothed_y = lerp(curr_y, ema_y, 0.3)
                    if abs(curr_x - smoothed_x) > 1 or abs(curr_y - smoothed_y) > 1:
                        pyautogui.moveTo(int(smoothed_x), int(smoothed_y))
                prev_pixel_x, prev_pixel_y = px, py

        if not right_hand_found:
            prev_pixel_x, prev_pixel_y = None, None

        elapsed = time.time() - start_time
        time.sleep(max(0, 1/target_fps - elapsed))

# Cập nhật frame
def update_click_frame(frame):
    global global_click_frame
    with global_click_lock:
        global_click_frame = frame.copy()

# Bắt đầu thread xử lý click
click_thread = threading.Thread(target=click_processing, daemon=True)
click_thread.start()

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không lấy được frame từ camera.")
            break

        now = time.time()
        if now - prev_time < 1/target_fps:
            continue
        prev_time = now

        update_click_frame(frame)

        gestures_result, handedness_result, hand_landmarks_result = process_frame(frame)
        annotated_image = frame.copy()

        for i in range(len(gestures_result)):
            if gestures_result[i]:
                print(f"[DEBUG] Tay {i} - Gesture: {gestures_result[i][0].category_name}")

        if hand_landmarks_result:
            for landmarks in hand_landmarks_result:
                if not hasattr(landmarks, 'landmark'):
                    temp = landmark_pb2.NormalizedLandmarkList()
                    for lm in landmarks:
                        new_lm = temp.landmark.add()
                        new_lm.x, new_lm.y, new_lm.z = lm.x, lm.y, lm.z
                        if hasattr(lm, 'visibility'): new_lm.visibility = lm.visibility
                        if hasattr(lm, 'presence'): new_lm.presence = lm.presence
                    mp_drawing.draw_landmarks(
                        annotated_image, temp,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(thickness=2)
                    )
                else:
                    mp_drawing.draw_landmarks(
                        annotated_image, landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(thickness=2)
                    )

        cv2.imshow("Hand Gesture Recognition", annotated_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
