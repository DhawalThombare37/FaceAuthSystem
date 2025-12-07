import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

def aspect_ratio(points, eye):
    p1, p2, p3, p4, p5, p6 = [points[i] for i in eye]
    return ((abs(p2[1] - p6[1]) + abs(p3[1] - p5[1])) /
            (2.0 * abs(p1[0] - p4[0])))

def detect_blink(frame, blink_thresh=0.25, frame_count=0):
    rgb = frame[:, :, ::-1]
    results = mp_face_mesh.process(rgb)

    blink = False

    if results.multi_face_landmarks:
        for lm in results.multi_face_landmarks:
            h, w, _ = frame.shape
            points = [(int(p.x*w), int(p.y*h)) for p in lm.landmark]

            left = aspect_ratio(points, LEFT_EYE)
            right = aspect_ratio(points, RIGHT_EYE)
            ear = (left + right) / 2

            if ear < blink_thresh:
                frame_count += 1
            else:
                if frame_count > 2:
                    blink = True
                frame_count = 0

    return blink, frame_count

