# TinyGPT: Aman's Personal AI Portfolio

A miniature LLM built from scratch using PyTorch, trained exclusively on Aman Rajbhar's professional portfolio and resume.

Uses a character-level **Transformer architecture** (the same math powering GPT-4 and Claude) in ~200 lines of pure Python, served via a **FastAPI** backend and a plain HTML frontend.

## 🚀 Features

- **Zero API Calls:** Matrix multiplications, attention, and backpropagation run entirely on your machine.
- **Custom Dataset:** Learns grammar and facts purely from Aman's portfolio data.
- **Multi-Head Self-Attention:** Implements the core $Q, K, V$ architecture for contextual understanding.
- **Temperature & Top-K Sampling:** Generation controls to tune creativity and coherence.
- **FastAPI Backend:** Serves the model via a REST API (`/api/generate`).
- **HTML Frontend:** A simple browser UI (`index.html`) to chat with the model locally.

## 📁 Project Structure

```
tinyLLM/
├── main.py           # Model architecture, training loop, and generate function
├── app.py            # FastAPI server — loads trained weights and exposes API
├── index.html        # Browser-based chat UI
├── render.yaml       # Render deployment config
├── aman_model.pt     # Saved model weights (generated after training, not committed)
└── requirements.txt  # Python dependencies
```

## 🛠️ Local Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Train the model** *(skip if `aman_model.pt` already exists)*

```bash
python main.py
```

This trains for 5000 steps and saves weights to `aman_model.pt`.

**3. Start the API server**

```bash
python -m uvicorn app:app --reload
```

Server runs at `http://localhost:8000`. Open that URL in your browser to use the chat UI.

## ☁️ Deploy to Render

**1. Push to GitHub**

```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/<your-username>/tinyLLM.git
git push -u origin main
```

> Do **not** commit `aman_model.pt` — the server trains automatically on first startup.

**2. Create a new Web Service on [Render](https://render.com)**

- Connect your GitHub repo
- Render auto-detects `render.yaml` and configures everything
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

**3. Done** — your app will be live at `https://tinygpt.onrender.com` (or your chosen name).

## 🔌 API

**POST** `/api/generate`

```json
{
  "prompt": "Aman Rajbhar is",
  "max_tokens": 150
}
```

**Response:**

```json
{
  "prompt": "Aman Rajbhar is",
  "generated_text": "Aman Rajbhar is a Software Engineer based in Mumbai..."
}
```

**GET** `/` — Health check

## ⚙️ Hyperparameters

| Parameter    | Value  |
|-------------|--------|
| `n_embd`    | 64     |
| `n_heads`   | 4      |
| `n_layers`  | 3      |
| `block_size`| 64     |
| `max_iters` | 5000   |
| `dropout`   | 0.1    |

## 📝 Notes

- If `aman_model.pt` is not found, `app.py` will warn you and the API will still start but generate poor output. Run `main.py` first.
- CORS is open (`allow_origins=["*"]`) for local development. Restrict this in production.
- The model runs on GPU automatically if CUDA is available, otherwise falls back to CPU.
