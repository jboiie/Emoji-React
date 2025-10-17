#!/usr/bin/env python3
"""
Real-time emoji display based on camera pose and facial expression detection,
including 'sideeye' for head turn and 'tongue out' for tongue with smile detection.
"""

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# Configuration
SMILE_THRESHOLD = 0.22  # Smile (mouth aspect ratio) is not primary here
HEAD_TURN_THRESHOLD = 0.04    # for nose offset on head turn
TONGUE_THRESHOLD = 0.04       # tongue tip distance
TEETH_BRIGHTNESS_THRESHOLD = 120  # tune for your camera/environment
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 450
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

# Load emoji images
try:
    smiling_emoji = cv2.imread("smile.jpg")
    straight_face_emoji = cv2.imread("plain.png")
    hands_up_emoji = cv2.imread("air.jpg")
    tongue_emoji = cv2.imread("tongue.jpg")
    sideeye_emoji = cv2.imread("sideye.jpeg") # ensure ".jpg" matches file you provided
    
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
        selected_emoji = straight_face_emoji
        emoji_name = "😐"

        # Check for hands up
        results_pose = pose.process(image_rgb)
        hands_detected = False
        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            if (left_wrist.y < left_shoulder.y) or (right_wrist.y < right_shoulder.y):
                current_state = "HANDS_UP"
                selected_emoji = hands_up_emoji
                emoji_name = "🙌"
                hands_detected = True

        if not hands_detected:
            results_face = face_mesh.process(image_rgb)
            teeth_visible = False
            tongue_out = False
            side_eye = False

            if results_face.multi_face_landmarks:
                face_landmarks = results_face.multi_face_landmarks[0].landmark

                # Smile with teeth detection - Estimate brightness of mouth region
                mouth_points = [13, 14, 78, 308, 82, 312, 87, 317]
                mouth_landmarks = np.array([
                    (face_landmarks[pt].x, face_landmarks[pt].y)
                    for pt in mouth_points
                ])
                img_h, img_w = frame.shape[:2]
                mouth_coords = (mouth_landmarks * np.array([img_w, img_h])).astype(np.int32)
                mouth_rect = cv2.boundingRect(mouth_coords)
                x, y, w, h = mouth_rect
                mouth_roi = frame[y:y+h, x:x+w]
                avg_intensity = 0
                if mouth_roi.size > 0:
                    mouth_gray = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
                    avg_intensity = mouth_gray.mean()
                teeth_visible = avg_intensity > TEETH_BRIGHTNESS_THRESHOLD

                # Tongue out detection (distance between tip and lower lip middle)
                tongue_tip = face_landmarks[16]
                lower_lip_mid = face_landmarks[14]
                tongue_dist = ((tongue_tip.x - lower_lip_mid.x)**2 + (tongue_tip.y - lower_lip_mid.y)**2)**0.5
                tongue_out = teeth_visible and (tongue_dist > TONGUE_THRESHOLD)

                # Head turn detection (sideeye)
                nose_tip = face_landmarks[1]
                left_eye_outer = face_landmarks[130]
                right_eye_outer = face_landmarks[359]
                face_center_x = (left_eye_outer.x + right_eye_outer.x) / 2
                nose_offset = nose_tip.x - face_center_x
                side_eye = abs(nose_offset) > HEAD_TURN_THRESHOLD

                # Priority: hands > tongue out+smile > sideeye > smile w/teeth > straight
                if tongue_out:
                    current_state = "TONGUE_OUT"
                    selected_emoji = tongue_emoji
                    emoji_name = "😛"
                elif side_eye:
                    current_state = "SIDE_EYE"
                    selected_emoji = sideeye_emoji
                    emoji_name = "🙄"
                elif teeth_visible:
                    current_state = "SMILING"
                    selected_emoji = smiling_emoji
                    emoji_name = "😊"
                else:
                    current_state = "STRAIGHT_FACE"
                    selected_emoji = straight_face_emoji
                    emoji_name = "😐"
            else:
                current_state = "NO_FACE"
                selected_emoji = blank_emoji
                emoji_name = "❓"

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
