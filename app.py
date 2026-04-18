from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import os

from main import model, generate_from_prompt, device, train_model

app = FastAPI(title="Aman's TinyGPT API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists('aman_model.pt'):
    model.load_state_dict(torch.load('aman_model.pt', map_location=device))
    model.eval()
    print("Loaded pre-trained model weights.")
else:
    print("No weights found. Training model now...")
    train_model()
    print("Training complete.")

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 150

@app.get("/")
def index():
    return FileResponse("index.html")

@app.post("/api/generate")
def generate_text(req: PromptRequest):
    result = generate_from_prompt(req.prompt, req.max_tokens)
    return {"prompt": req.prompt, "generated_text": result}