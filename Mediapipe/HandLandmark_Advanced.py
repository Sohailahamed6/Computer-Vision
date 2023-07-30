import mediapipe as mp
import cv2 as cv
import numpy as np

mp_drawing=mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def fingerprint(hand):
    coordsth = coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x, hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y)),
                [640, 480]).astype(int))
    coordsind = coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y)),
                [640, 480]).astype(int))
    coordsmid = coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x, hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y)),
                [640, 480]).astype(int))
    coordsr = coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x, hand.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y)),
                [640, 480]).astype(int))
    coordsp = coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.PINKY_TIP].x, hand.landmark[mp_hands.HandLandmark.PINKY_TIP].y)),
                [640, 480]).astype(int))
    return coordsth,coordsind,coordsmid,coordsr,coordsp
def get_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            # Process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))

            # Extract Coordinates
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                [640, 480]).astype(int))

            output = text, coords

    return output

cap = cv.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        # BGR 2 RGB
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        # Detections
        #print(results.multi_hand_landmarks)

        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):

                # Render left or right detection
                if get_label(num, hand, results):
                    text, coord = get_label(num, hand, results)
                    cv.putText(image, text, coord, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
                if fingerprint(hand):
                    coordsth,coordsind,coordsmid,coordsr,coordsp = fingerprint(hand)
                    cv.putText(image, "Thumb", coordsth, cv.FONT_HERSHEY_SIMPLEX, 1, (250, 255, 255), 2, cv.LINE_AA)
                    cv.putText(image, "Index", coordsind, cv.FONT_HERSHEY_SIMPLEX, 1, (250, 255, 255), 2, cv.LINE_AA)
                    cv.putText(image, "Middle", coordsmid, cv.FONT_HERSHEY_SIMPLEX, 1, (250, 255, 255), 2, cv.LINE_AA)
                    cv.putText(image, "Ring", coordsr, cv.FONT_HERSHEY_SIMPLEX, 1, (250, 255, 255), 2, cv.LINE_AA)
                    cv.putText(image, "Pinky", coordsp, cv.FONT_HERSHEY_SIMPLEX, 1, (250, 255, 255), 2, cv.LINE_AA)

        cv.imshow('Hand Tracking', image)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()

