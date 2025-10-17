#!/usr/bin/env python3
"""
Real-time emoji display based on camera pose and facial expression detection,
including 'sideeye' for head turn and 'tongue out' for tongue detection.
"""

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Configuration
SMILE_THRESHOLD = 0.2
HEAD_TURN_THRESHOLD = 0.02  # Tune this for your camera framing
TONGUE_THRESHOLD = 0.05     # Ratio for tongue out detection
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 450
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

# Load emoji images
try:
    smiling_emoji = cv2.imread("smile.jpg")
    straight_face_emoji = cv2.imread("plain.png")
    hands_up_emoji = cv2.imread("air.jpg")
    tongue_emoji = cv2.imread("tongue.jpg")
    sideeye_emoji = cv2.imread("sideye.jpeg")
    
    if smiling_emoji is None:
        raise FileNotFoundError("smile.jpg not found")
    if straight_face_emoji is None:
        raise FileNotFoundError("plain.png not found")
    if hands_up_emoji is None:
        raise FileNotFoundError("air.jpg not found")
    if tongue_emoji is None:
        raise FileNotFoundError("tongue.jpg not found")
    if sideeye_emoji is None:
        raise FileNotFoundError("sideye.jpg not found")
    
    # Resize emojis
    smiling_emoji = cv2.resize(smiling_emoji, EMOJI_WINDOW_SIZE)
    straight_face_emoji = cv2.resize(straight_face_emoji, EMOJI_WINDOW_SIZE)
    hands_up_emoji = cv2.resize(hands_up_emoji, EMOJI_WINDOW_SIZE)
    tongue_emoji = cv2.resize(tongue_emoji, EMOJI_WINDOW_SIZE)
    sideeye_emoji = cv2.resize(sideeye_emoji, EMOJI_WINDOW_SIZE)
    
except Exception as e:
    print("Error loading emoji images!")
    print(f"Details: {e}")
    exit()

blank_emoji = np.zeros((EMOJI_WINDOW_SIZE[1], EMOJI_WINDOW_SIZE[0], 3), dtype=np.uint8)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cv2.namedWindow('Emoji Output', cv2.WINDOW_NORMAL)
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Feed', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow('Emoji Output', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.moveWindow('Camera Feed', 100, 100)
cv2.moveWindow('Emoji Output', WINDOW_WIDTH + 150, 100)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        current_state = "STRAIGHT_FACE"

        # Check for hands up
        results_pose = pose.process(image_rgb)
        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            if (left_wrist.y < left_shoulder.y) or (right_wrist.y < right_shoulder.y):
                current_state = "HANDS_UP"

        # Check facial features if hands not up
        selected_emoji = straight_face_emoji
        emoji_name = "😐"
        if current_state != "HANDS_UP":
            results_face = face_mesh.process(image_rgb)
            if results_face.multi_face_landmarks:
                face_landmarks = results_face.multi_face_landmarks[0].landmark
                # Smile detection
                left_corner = face_landmarks[291]
                right_corner = face_landmarks[61]
                upper_lip = face_landmarks[13]
                lower_lip = face_landmarks[14]

                mouth_width = ((right_corner.x - left_corner.x)**2 + (right_corner.y - left_corner.y)**2)**0.5
                mouth_height = ((lower_lip.x - upper_lip.x)**2 + (lower_lip.y - upper_lip.y)**2)**0.5
                mouth_aspect_ratio = mouth_height / mouth_width if mouth_width > 0 else 0

                # --- Tongue out detection ---
                tongue_tip = face_landmarks[16]       # Approximated for tongue
                lower_lip_mid = face_landmarks[14]
                tongue_dist = ((tongue_tip.x - lower_lip_mid.x)**2 + (tongue_tip.y - lower_lip_mid.y)**2)**0.5

                # --- Head turn detection (sideeye) ---
                nose_tip = face_landmarks[1]
                left_eye_outer = face_landmarks[130]
                right_eye_outer = face_landmarks[359]
                # X diff between nose and center line (rough, for frontal webcam)
                face_center_x = (left_eye_outer.x + right_eye_outer.x) / 2
                nose_offset = nose_tip.x - face_center_x

                # Priority: hands > tongue out > head turn > smile > straight
                if tongue_dist > TONGUE_THRESHOLD:
                    current_state = "TONGUE_OUT"
                    selected_emoji = tongue_emoji
                    emoji_name = "😛"
                elif abs(nose_offset) > HEAD_TURN_THRESHOLD:
                    current_state = "SIDE_EYE"
                    selected_emoji = sideeye_emoji
                    emoji_name = "🙄"
                elif mouth_aspect_ratio > SMILE_THRESHOLD:
                    current_state = "SMILING"
                    selected_emoji = smiling_emoji
                    emoji_name = "😊"
                else:
                    current_state = "STRAIGHT_FACE"
                    selected_emoji = straight_face_emoji
                    emoji_name = "😐"
            else:
                # If no face, just blank
                current_state = "NO_FACE"
                selected_emoji = blank_emoji
                emoji_name = "❓"
        else:
            selected_emoji = hands_up_emoji
            emoji_name = "🙌"

        camera_frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        cv2.putText(camera_frame_resized, f'STATE: {current_state} {emoji_name}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(camera_frame_resized, 'Press "q" to quit', (10, WINDOW_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Camera Feed', camera_frame_resized)
        cv2.imshow('Emoji Output', selected_emoji)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
