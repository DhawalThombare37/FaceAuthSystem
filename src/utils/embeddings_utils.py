import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load FaceNet once globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def preprocess_face(face):
    face = cv2.resize(face, (160, 160))
    face = face / 255.0
    face = torch.tensor(face, dtype=torch.float32).permute(2, 0, 1)
    return face.unsqueeze(0).to(device)

def get_embedding(face):
    face_tensor = preprocess_face(face)
    with torch.no_grad():
        emb = model(face_tensor).cpu().numpy().flatten()
    return emb / np.linalg.norm(emb)

def load_all_embeddings(emb_dir="embeddings"):
    files = [f for f in os.listdir(emb_dir) if f.endswith(".npy")]
    if not files:
        print("⚠️ No embeddings found! Run enrollment first.")
        return None, None

    embeddings, names = [], []
    for f in files:
        emb = np.load(os.path.join(emb_dir, f))
        embeddings.append(emb)
        names.append(os.path.splitext(f)[0])

    return np.vstack(embeddings), names

def match_embedding(live_emb, db_embs, names, threshold=0.7):
    sims = cosine_similarity(live_emb.reshape(1, -1), db_embs)[0]
    idx = np.argmax(sims)
    if sims[idx] >= threshold:
        return names[idx], sims[idx]
    return None, sims[idx]

