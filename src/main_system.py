import cv2
import mediapipe as mp
import torch
from facenet_pytorch import InceptionResnetV1
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os

# -------------------- Setup --------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

mp_face = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

detector = mp_face.FaceDetection(min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

def get_embedding(face_img):
    face_img = cv2.resize(face_img, (160, 160))
    face_img = torch.tensor(face_img / 255., dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(face_img).cpu().numpy().flatten()
    return emb / np.linalg.norm(emb)

def log_event(user_id, action):
    file = "logs.csv"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = pd.DataFrame([[user_id, action, now]])
    if os.path.exists(file):
        row.to_csv(file, mode='a', header=False, index=False)
    else:
        row.to_csv(file, header=["user_id", "action", "timestamp"], index=False)
    print(f"✅ Logged {action} for {user_id} at {now}")

def aspect_ratio(landmarks, eye):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye]
    return ((abs(p2[1]-p6[1])+abs(p3[1]-p5[1])) / (2.0*abs(p1[0]-p4[0])))

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

# -------------------- Load embeddings --------------------
# -------------------- Load embeddings --------------------
import os
import glob
import numpy as np

EMBEDDINGS_DIR = os.path.join(os.path.dirname(__file__), "embeddings")

npy_files = glob.glob(os.path.join(EMBEDDINGS_DIR, "*.npy"))

if not npy_files:
    print("⚠️ No embeddings found! Run enroll.py first.")
    raise SystemExit

embeddings = []
user_ids = []  # <- this is what you'll use later

for path in npy_files:
    emb = np.load(path)
    embeddings.append(emb)
    # filename is the label, e.g. "john.npy" -> "john"
    user_id = os.path.splitext(os.path.basename(path))[0]
    user_ids.append(user_id)

# make embeddings a 2D array (n_users, emb_dim)
embeddings = np.vstack(embeddings)

print(f"Loaded {len(user_ids)} embeddings:", user_ids)



# -------------------- IN/OUT state --------------------
state_file = "states.csv"
if os.path.exists(state_file):
    states = pd.read_csv(state_file).set_index("user_id")["state"].to_dict()
else:
    states = {}

def update_state(user, action):
    states[user] = action
    pd.DataFrame(list(states.items()), columns=["user_id","state"]).to_csv(state_file, index=False)

# -------------------- Main Loop --------------------
cap = cv2.VideoCapture(0)

blink_count = 0
frame_count = 0
blink_thresh = 0.25
recognized_user = None
liveness_passed = False

print("📷 System Running – Q quit | O entry | X exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)
    mesh_results = face_mesh.process(rgb)

    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
            x2, y2 = x1 + int(bbox.width * w), y1 + int(bbox.height * h)

            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                emb = get_embedding(face).reshape(1, -1)
                sims = cosine_similarity(emb, embeddings)[0]
                best_idx = np.argmax(sims)

                if sims[best_idx] > 0.7:
                    recognized_user = user_ids[best_idx]
                    cv2.putText(frame, f"{recognized_user}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                else:
                    recognized_user = None
                    cv2.putText(frame, "Unknown", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    # -------------------- Liveness (Blink) --------------------
    if mesh_results.multi_face_landmarks:
        for lm in mesh_results.multi_face_landmarks:
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
                    liveness_passed = True
                    print("✅ Liveness confirmed (blink)")
                frame_count = 0

    # -------------------- Instructions --------------------
    if recognized_user and liveness_passed:
        cv2.putText(frame, "Press O=Entry | X=Exit", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('o') and recognized_user and liveness_passed:
        if states.get(recognized_user) == "IN":
            print(f"⚠️ {recognized_user} already IN – must EXIT first")
        else:
            log_event(recognized_user, "IN")
            update_state(recognized_user, "IN")

    elif key == ord('x') and recognized_user and liveness_passed:
        if states.get(recognized_user) != "IN":
            print(f"⚠️ {recognized_user} not IN – cannot EXIT")
        else:
            log_event(recognized_user, "OUT")
            update_state(recognized_user, "OUT")

    elif key == ord('q'):
        break

    cv2.imshow("Face Auth System", frame)

cap.release()
cv2.destroyAllWindows()

