from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import os
import threading
import time

from main import model, generate_from_prompt, device, train_model

app = FastAPI(title="mannLLM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'aman_model.pt')

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("mannLLM: Loaded pre-trained weights.")
    except RuntimeError as e:
        print(f"mannLLM: Weight mismatch — starting with random weights. Trigger /api/train to retrain. ({e})")
else:
    print("mannLLM: No weights found. Please train via /api/train.")

train_status = {"running": False, "message": "Idle"}
cancel_flag  = {"cancel": False}

from typing import List, Optional

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 150

class ChatMessage(BaseModel):
    role: str   # 'user' or 'bot'
    text: str

class TrainRequest(BaseModel):
    chat_history: Optional[List[ChatMessage]] = []

@app.get("/")
def index():
    return FileResponse("index.html")

@app.post("/api/generate")
def generate_text(req: PromptRequest):
    if train_status["running"]:
        return {"prompt": req.prompt, "generated_text": "Model is currently training. Please wait and try again shortly."}
    result = generate_from_prompt(req.prompt, req.max_tokens)
    return {"prompt": req.prompt, "generated_text": result}

@app.post("/api/train")
def start_training(req: TrainRequest = TrainRequest()):
    if train_status["running"]:
        return {"status": "already_running", "message": "Training is already in progress."}

    # Build extra training text from chat history
    extra = ""
    if req.chat_history:
        pairs = []
        history = req.chat_history
        for i in range(len(history) - 1):
            if history[i].role == 'user' and history[i+1].role == 'bot':
                pairs.append(f"User: {history[i].text}\nBot: {history[i+1].text}")
        if pairs:
            extra = "\n\n".join(pairs)
            print(f"mannLLM: Received {len(pairs)} chat pairs for training.")

    def run():
        cancel_flag["cancel"] = False
        train_status["running"] = True
        train_status["message"] = "Training in progress..."
        try:
            train_model(cancel_flag, extra_text=extra)
            if cancel_flag["cancel"]:
                train_status["message"] = "Training cancelled."
            else:
                model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                model.eval()
                train_status["message"] = "Training complete. Model reloaded."
        except Exception as e:
            train_status["message"] = f"Training failed: {str(e)}"
        finally:
            train_status["running"] = False
            cancel_flag["cancel"]   = False

    threading.Thread(target=run, daemon=True).start()
    return {"status": "started", "message": "Training started in background."}

@app.post("/api/train/cancel")
def cancel_training():
    if not train_status["running"]:
        return {"status": "not_running", "message": "No training in progress."}
    cancel_flag["cancel"] = True
    train_status["message"] = "Cancelling..."
    return {"status": "cancelling", "message": "Cancel signal sent."}

@app.get("/api/train/status")
def training_status():
    return train_status