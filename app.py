from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch

# Import the model and functions from your main script
from main import model, generate_from_prompt, device

app = FastAPI(title="Aman's TinyGPT API")

# Allow requests from your Angular/Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change this to your Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained weights when the server starts
try:
    model.load_state_dict(torch.load('aman_model.pt', map_location=device))
    model.eval()
    print("Successfully loaded pre-trained model weights.")
except FileNotFoundError:
    print("WARNING: aman_model.pt not found. Run main.py first to train the model!")

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 150

@app.post("/api/generate")
def generate_text(req: PromptRequest):
    # Generate text using the function from main.py
    result = generate_from_prompt(req.prompt, req.max_tokens)
    return {
        "prompt": req.prompt,
        "generated_text": result
    }

@app.get("/")
def health_check():
    return {"status": "Aman's TinyGPT API is running!"}