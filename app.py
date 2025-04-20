from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
import base64
import io
from keras.models import load_model
import mediapipe as mp
from youtube_search import YoutubeSearch

# Load model and labels
model = load_model("model.h5")
label = np.load("labels.npy")

# Initialize mediapipe
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# FastAPI app setup
app = FastAPI()

# Allow frontend CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Data model
class ImageData(BaseModel):
    image_data: str  # base64 encoded image string

# Helper: Get YouTube URL from a query
def get_youtube_video(song_query):
    results = YoutubeSearch(song_query, max_results=1).to_dict()
    if results:
        video_id = results[0]['id']
        return f"https://www.youtube.com/embed/{video_id}"
    return None

# Endpoint: Detect Emotion
@app.post("/detect_emotion")
async def detect_emotion(data: ImageData):
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(data.image_data.split(",")[1])
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Process image with mediapipe
        res = holis.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            lst = np.array(lst).reshape(1, -1)
            pred = label[np.argmax(model.predict(lst))]
            return {"emotion": pred}
        else:
            return {"emotion": "unknown"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

# Endpoint: Recommend Song
@app.get("/recommend_song")
async def recommend_song(emotion: str):
    emotion_to_keywords = {
        "happy": "happy mood song",
        "sad": "sad song",
        "angry": "angry rock music",
        "shocked": "surprising instrumental",
        "rock": "rock anthem"
    }

    query = emotion_to_keywords.get(emotion.lower(), None)
    if query:
        video_url = get_youtube_video(query)
        if video_url:
            return {"song": video_url}
        else:
            return {"song": "https://www.youtube.com"}  # fallback
    else:
        return {"song": "https://www.youtube.com"}
