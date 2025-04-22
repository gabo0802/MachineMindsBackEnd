
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load
import pandas as pd
import os

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the model once on startup
model_path = os.path.join(os.path.dirname(__file__), 'difficulty_ai_weights_REBUILT.joblib')
model = load(model_path)

# Define the expected input model
class GameData(BaseModel):
    currentDifficulty: float
    currentPlayerLives: float
    levelsBeat: float
    playerLifeTimer: float
    totalEnemiesKilled: float
    totalPoints: float

@app.get("/")
def read_root():
    return {"message": "ML API is live"}

@app.post("/predict")
async def predict_difficulty(data: GameData):
    try:
        x_value = pd.DataFrame([{
            'currentDifficulty': data.currentDifficulty,
            'currentPlayerLives': data.currentPlayerLives,
            'levelsBeat': data.levelsBeat,
            'playerLifeTimer': data.playerLifeTimer,
            'totalEnemiesKilled': data.totalEnemiesKilled,
            'totalPoints': data.totalPoints
        }])

        prediction = model.predict(x_value)[0]
        return {"prediction": int(prediction)}

    except Exception as e:
        return {"error": str(e)}
