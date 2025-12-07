# enroll_user.py
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
import torch
from torchvision import transforms
import mediapipe as mp
import os

# Load facenet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# Mediapipe face detector
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)

# Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def get_embedding(img_tensor):
    with torch.no_grad():
        emb = model(img_tensor.unsqueeze(0).to(device))
    return emb.cpu().numpy()[0]

# ---- MAIN ----
name = input("Enter user name for enrollment: ")
os.makedirs("embeddings", exist_ok=True)

cap = cv2.VideoCapture(0)
print("Press 'c' to capture face and enroll, or 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)

    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
            x2, y2 = x1 + int(bbox.width * w), y1 + int(bbox.height * h)
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.imshow("Enrollment", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        if results.detections:
            face_tensor = transform(face_crop)
            emb = get_embedding(face_tensor)
            np.save(f"embeddings/{name.lower()}.npy", emb)
            print(f"✅ User {name} enrolled and embedding saved!")
            break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

