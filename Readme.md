# mannLLM — Aman's Personal AI

A miniature LLM built from scratch using PyTorch, trained exclusively on Aman Rajbhar's professional portfolio, resume, and real chat history.

Uses a character-level **Transformer architecture** in ~200 lines of pure Python, served via a **FastAPI** backend and a glassmorphism chat UI.

🔗 **Live:** [mannllm.onrender.com](https://mannllm.onrender.com)

## 🚀 Features

- **Zero API Calls** — Matrix multiplications, attention, and backpropagation run entirely on your machine.
- **Q&A Training Data** — Trained on 50+ exact question-answer pairs about Aman's skills, projects, and experience.
- **Chat History Training** — Real conversations stored in localStorage are fed back into training to improve the model over time.
- **On-Demand Training** — Trigger retraining from the UI via the ⚙️ settings menu without restarting the server.
- **Cancel Training** — Stop training mid-way with a cancel button; old weights remain intact.
- **Multi-Head Self-Attention** — Implements the core $Q, K, V$ Transformer architecture.
- **Temperature & Top-K Sampling** — Controls generation creativity and coherence.
- **Glassmorphism Chat UI** — Apple-style dark/light theme chat interface with typing indicators.
- **FastAPI Backend** — REST API with `/api/generate`, `/api/train`, `/api/train/status`, `/api/train/cancel`.

## 📁 Project Structure

```
tinyLLM/
├── main.py            # Model architecture, Q&A dataset, training loop, generate function
├── app.py             # FastAPI server — loads weights, exposes API endpoints
├── index.html         # Glassmorphism chat UI with dark/light theme
├── render.yaml        # Render deployment config
├── .python-version    # Python version pin for Render
├── aman_model.pt      # Saved model weights (committed, skip training on startup)
└── requirements.txt   # Python dependencies
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

Trains for 5000 steps and saves weights to `aman_model.pt`.

**3. Start the server**

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in your browser.

> Use `python -m uvicorn` instead of `uvicorn` directly on Windows if uvicorn is not in PATH.

## ☁️ Deploy to Render

**1. Push to GitHub** (commit `aman_model.pt` so Render skips training on startup)

```bash
git add .
git commit -m "initial commit"
git remote add origin https://github.com/<your-username>/tinyLLM.git
git push -u origin main
```

**2. Create a Web Service on [Render](https://render.com)**

- Connect your GitHub repo
- Render auto-detects `render.yaml`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

**3. Done** — live at `https://mannllm.onrender.com`

## 🔌 API

**POST** `/api/generate`
```json
{ "prompt": "who is aman", "max_tokens": 150 }
```

**POST** `/api/train`
```json
{ "chat_history": [{ "role": "user", "text": "hey" }, { "role": "bot", "text": "Hey! What would you like to know about Aman?" }] }
```

**GET** `/api/train/status`

**POST** `/api/train/cancel`

## ⚙️ Hyperparameters

| Parameter    | Value |
|-------------|-------|
| `n_embd`    | 64    |
| `n_heads`   | 4     |
| `n_layers`  | 3     |
| `block_size`| 64    |
| `max_iters` | 5000  |
| `dropout`   | 0.1   |

## 📝 Notes

- `aman_model.pt` is committed to the repo so Render loads weights instantly on startup — no training delay.
- Training can be triggered anytime from the ⚙️ menu in the UI. Chat history from localStorage is automatically included.
- CORS is open (`allow_origins=["*"]`) for local dev. Restrict in production.
- The model runs on GPU if CUDA is available, otherwise falls back to CPU.
- On Render free tier, the service sleeps after inactivity — first request may take ~30s to wake up.

## 👤 Author

Built by [Aman Rajbhar](https://amnnrajbhar.github.io/info/) — Software Engineer at Clover Infotech, Mumbai.
