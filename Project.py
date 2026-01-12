import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands=mp_hands.Hands(min_detection_confidence=0.7,min_tracking_confidence=0.7)
mp_draw=mp.solutions.drawing_utils

cap=cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Hand tracking started. Press 'q' to exit.")


drawing_points = []

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    gesture = "No hand detected"

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label
            landmarks = hand_landmarks.landmark
            tip_ids = [4, 8, 12, 16, 20]
            pip_ids = [2, 6, 10, 14, 18]
            extended = 0

            # Detect if fingers are extended
            if abs(landmarks[tip_ids[0]].x - landmarks[pip_ids[0]].x) > 0.004:
                extended += 1

            for i in range(1, 5):
                if landmarks[tip_ids[i]].y < landmarks[pip_ids[i]].y:
                    extended += 1

            if extended >= 4:
                gesture = "Open"
            elif extended <= 1:
                gesture = "Closed"
            else:
                gesture = "Partial"

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingertips_ids = [4, 8, 12, 16, 20]
            raised_fingers = 0  

            for tip_id in fingertips_ids:
                lm = hand_landmarks.landmark[tip_id]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 10, (255, 0, 0), cv2.FILLED)
                cv2.putText(frame, str(tip_id), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                if lm.y < hand_landmarks.landmark[0].y:  
                    raised_fingers += 1

                if tip_id == 8: 
                    drawing_points.append((x, y))

            if raised_fingers == 0:
                drawing_points = []

            wrist = hand_landmarks.landmark[0]
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
            cv2.putText(frame, f'{hand_label} Hand: {gesture}', (wrist_x - 40, wrist_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    for i in range(1, len(drawing_points)):
        cv2.line(frame, drawing_points[i - 1], drawing_points[i], (0, 0, 255), 3)

    status_color = (0, 255, 0) if gesture in ["Open", "Closed"] else (0, 165, 255)
    cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    cv2.imshow("Hand Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()