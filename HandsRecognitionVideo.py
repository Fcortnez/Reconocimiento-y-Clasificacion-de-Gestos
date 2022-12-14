import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:

    while True:
        ret, frame = cap.read()
        if ret == False:
            break

        height, width, _ = frame.shape
        frame =cv2.flip(frame,1)
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        resultes = hands.process(frame_RGB)

        if resultes.multi_hand_landmarks is not None:
            for hand_landmarks in resultes.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(0,255,0), thickness=5, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(205,50,255),thickness=2))


        cv2.imshow("Reconocimiento de Manos AulaVirtualRV",frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()