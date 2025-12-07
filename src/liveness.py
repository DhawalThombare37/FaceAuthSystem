import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

def aspect_ratio(landmarks, eye):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye]
    return ((abs(p2[1]-p6[1])+abs(p3[1]-p5[1])) / (2.0*abs(p1[0]-p4[0])))

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

blink_count = 0
blink_thresh = 0.25
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for lm in results.multi_face_landmarks:
            h, w, _ = frame.shape
            points = [(int(p.x*w), int(p.y*h)) for p in lm.landmark]

            left_ratio = aspect_ratio(points, LEFT_EYE)
            right_ratio = aspect_ratio(points, RIGHT_EYE)
            avg_ratio = (left_ratio + right_ratio) / 2

            if avg_ratio < blink_thresh:
                frame_count += 1
            else:
                if frame_count > 2:
                    blink_count += 1
                    print("Blink detected! Total:", blink_count)
                frame_count = 0

    cv2.imshow("Liveness Check", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

