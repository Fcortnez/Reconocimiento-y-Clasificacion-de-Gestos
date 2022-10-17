import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
    """image = cv2.imread("h1.jpg")"""
    image = cv2.imread("C:\muestra\h2.jpeg")
    height, width, _ = image.shape
    image = cv2.flip(image, 1)
    imgresize = cv2.resize(image, (520, 1200))  # cambiar tamaño img

    imageRGB = cv2.cvtColor(imgresize, cv2.COLOR_BGR2RGB)

    resultes = hands.process(imageRGB)

    # HANDEDNESS - Datos de que mano nos detecta
    #print("Handedness: ", resultes.multi_handedness)
    # HAND LANDMARKS
    #print("Hand landmarks: ", resultes.multi_hand_landmarks)

    if resultes.multi_hand_landmarks is not None:

        # Dibujando los puntos y conexiones con mediapipe
        for hand_landmarks in resultes.multi_hand_landmarks:
            #print(hand_landmarks)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255,255,0), thickness=20, circle_radius=5),
                                      mp_drawing.DrawingSpec(color=(205,2,255),thickness=10))


    image = cv2.flip(image, 1)
    imgresize = cv2.resize(image, (520, 700)) #cambiar tamaño img
    #cv2.imshow("Image", image)

    cv2.imshow("Reconocimiento de Manos AulaVirtualRV", imgresize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


