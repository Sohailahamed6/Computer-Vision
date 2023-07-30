import mediapipe as mp
import cv2 as cv

mp_drawing=mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.3,min_tracking_confidence=0.3) as holistic:
    while cap.isOpened():
        ret,frame=cap.read()
        image=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        results = holistic.process(image)
        print(results.pose_landmarks)

        image=cv.cvtColor(image,cv.COLOR_RGB2BGR)

        #face
        mp_drawing.draw_landmarks(image,results.face_landmarks,mp_holistic.FACEMESH_TESSELATION)
        #lefthand
        mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
        #righthand
        mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
        #pose
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        #image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        cv.imshow('Face',image )

        if cv.waitKey(10) & 0xFF == ord('q'):
            break;

cap.release()
cv.destroyAllWindows()

