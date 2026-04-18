# ============================================================
# mannLLM — TRAINED ON AMAN'S PERSONAL INFO
# A character-level Transformer built from scratch
# ============================================================

import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# ============================================================
# SECTION 1: THE DATASET (Extracted from Portfolio HTML)
# ============================================================

qa = """
User: hey
Bot: Hey! What would you like to know about Aman?

User: hi
Bot: Hi there! What would you like to know about Aman?

User: hello
Bot: Hello! What would you like to know about Aman?

User: hey there
Bot: Hey there! What would you like to know about Aman?

User: who are you
Bot: I am mannLLM, Aman Rajbhar's personal AI assistant. Ask me anything about Aman!

User: what is mannllm
Bot: mannLLM is a miniature language model built from scratch using PyTorch, trained exclusively on Aman Rajbhar's portfolio and resume data.

User: who is aman
Bot: Aman Rajbhar is a Software Engineer based in Mumbai, Maharashtra, India. He specializes in Angular and ASP.NET web development.

User: who is aman rajbhar
Bot: Aman Rajbhar is a Software Engineer from Mumbai, Maharashtra, India. He works at Clover Infotech and has expertise in Angular, TypeScript, and ASP.NET.

User: tell me about aman
Bot: Aman Rajbhar is a Software Engineer from Mumbai, India. He specializes in Angular and ASP.NET, has built many projects, and currently works at Clover Infotech since December 2024.

User: where does aman work
Bot: Aman currently works at Clover Infotech Private Limited as a Software Engineer since December 2024.

User: where did aman work before
Bot: Aman previously worked at Benchmark Computer Solutions Limited as a Junior Software Developer from August 2022 to December 2024.

User: what does aman do at clover infotech
Bot: At Clover Infotech, Aman develops high-performance web applications using Angular, TypeScript, and RxJS, and designs scalable microservices and REST APIs.

User: what did aman do at benchmark
Bot: At Benchmark Computer Solutions, Aman built dynamic web apps using Angular, ASP.NET, and MVC.

User: what is aman experience
Bot: Aman has over 2 years of experience. He worked at Benchmark Computer Solutions from August 2022 to December 2024, and currently works at Clover Infotech since December 2024.

User: what does aman do
Bot: Aman is a Software Engineer who builds high-performance web applications using Angular, TypeScript, RxJS, and ASP.NET. He also designs scalable REST APIs and microservices.

User: what are aman skills
Bot: Aman is skilled in Angular, ASP.NET MVC, .NET Core, React, Node.js, JavaScript, TypeScript, C#, Python, HTML, CSS, and SQL.

User: what languages does aman know
Bot: Aman knows JavaScript, TypeScript, C#, Python, HTML, CSS, and SQL.

User: what frameworks does aman use
Bot: Aman works with Angular, ASP.NET MVC, .NET Core, React, and Node.js.

User: what databases does aman know
Bot: Aman is experienced with SQL Server, Firebase Realtime Database, MySQL, and MongoDB.

User: what tools does aman use
Bot: Aman uses Visual Studio, Git, GitHub, XAMPP, and Crystal Reports in his workflow.

User: what methodologies does aman follow
Bot: Aman applies Agile methodologies, Scrum, and CI/CD practices in his workflow.

User: what projects has aman built
Bot: Aman has built a GenAI Web App, Media Upload Web App using AWS S3, HRMS Portal, LMS Portal, Digital Loan Process App, Property Seller platform, Weather Web App, and his personal Portfolio Website.

User: tell me about aman projects
Bot: Aman has built a GenAI chatbot using Ionic and Google Generative AI, an HRMS Portal with Angular and Node.js, an LMS Portal with Firebase, a Digital Loan App with MongoDB, and a Weather Web App using OpenWeather API.

User: what is the genai project
Bot: Aman built a GenAI Web App, an AI-powered chatbot using Ionic, Angular, and Google Generative AI.

User: what is the hrms project
Bot: Aman built an HRMS Portal, a Human Resource Management System using Angular, Node.js, and Tailwind CSS.

User: what is the lms project
Bot: Aman built an LMS Portal, a Learning Management System using Angular and Firebase.

User: what is the media upload project
Bot: Aman built a Media Upload Web App utilizing AWS S3 Buckets and Angular for file storage and management.

User: what is the loan app project
Bot: Aman built a Digital Loan Process App using Angular, Node.js, and MongoDB.

User: what is the weather app project
Bot: Aman built a Weather Web App for real-time weather updates using the OpenWeather API.

User: what is the portfolio website project
Bot: Aman built his personal Portfolio Website using Angular, TypeScript, SCSS, and Bootstrap.

User: what is aman education
Bot: Aman graduated with a B.Sc. in Computer Science from the University of Mumbai in May 2022. He is currently pursuing an MCA from IGNOU since June 2022.

User: where did aman study
Bot: Aman studied at the University of Mumbai where he completed his B.Sc. in Computer Science. He is also pursuing an MCA from IGNOU.

User: what is aman degree
Bot: Aman holds a B.Sc. in Computer Science from the University of Mumbai and is pursuing an MCA from IGNOU.

User: does aman have certifications
Bot: Yes! Aman holds a certification in Advanced Angular from Great Learning and a certification in ASP.NET MVC Development.

User: what certifications does aman have
Bot: Aman holds a certification in Advanced Angular from Great Learning and a certification in ASP.NET MVC Development.

User: how to contact aman
Bot: You can contact Aman via email at amanrajbhar289@gmail.com.

User: what is aman email
Bot: Aman's email address is amanrajbhar289@gmail.com.

User: where is aman from
Bot: Aman is from Mumbai, Maharashtra, India.

User: what is aman location
Bot: Aman is based in Mumbai, Maharashtra, India.

User: is aman good at angular
Bot: Yes! Aman is an expert in Angular. He has built multiple production apps using Angular, TypeScript, RxJS, and holds an Advanced Angular certification from Great Learning.

User: does aman know python
Bot: Yes, Aman knows Python and has used it in building machine learning projects including mannLLM.

User: does aman know react
Bot: Yes, Aman is skilled in React as part of his frontend development expertise.

User: does aman know aws
Bot: Yes, Aman has worked with AWS S3 Buckets in his Media Upload Web App project.

User: what is aman github
Bot: You can find Aman's projects on GitHub at github.com/amnnrajbhar.

User: what is aman portfolio
Bot: You can view Aman's portfolio at https://amnnrajbhar.github.io/info/

User: thanks
Bot: You're welcome! Feel free to ask anything else about Aman.

User: thank you
Bot: Happy to help! Let me know if you have more questions about Aman.

User: bye
Bot: Goodbye! Feel free to come back anytime to learn more about Aman.

User: ok
Bot: Sure! Let me know if you have any questions about Aman.

User: cool
Bot: Glad you think so! Ask me anything about Aman's skills, projects, or experience.

User: nice
Bot: Thanks! Feel free to ask me anything about Aman.

User: great
Bot: Awesome! What else would you like to know about Aman?

User: what is your name
Bot: I am mannLLM, Aman Rajbhar's personal AI assistant.

User: who made you
Bot: I was built by Aman Rajbhar using PyTorch from scratch.

User: how were you built
Bot: I am a character-level Transformer model built from scratch using PyTorch, trained exclusively on Aman Rajbhar's portfolio data.
"""

text = (qa.strip() + "\n\n") * 120

# ============================================================
# SECTION 2: HYPERPARAMETERS
# ============================================================

batch_size  = 16
block_size  = 64
max_iters   = 5000
eval_every  = 500
lr          = 3e-4
n_embd      = 64
n_heads     = 4
n_layers    = 3
dropout     = 0.1
device      = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================
# SECTION 3: TOKENIZER
# ============================================================

chars      = sorted(set(text))
vocab_size = len(chars)
stoi       = {c: i for i, c in enumerate(chars)}
itos       = {i: c for i, c in enumerate(chars)}
encode     = lambda s: [stoi[c] for c in s]
decode     = lambda l: ''.join([itos[i] for i in l])

# ============================================================
# SECTION 4: DATA SPLIT
# ============================================================

data       = torch.tensor(encode(text), dtype=torch.long)
n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

def get_batch(split):
    d  = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x  = torch.stack([d[i:i+block_size] for i in ix])
    y  = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ============================================================
# SECTIONS 5-9: MODEL ARCHITECTURE
# ============================================================

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.k    = nn.Linear(n_embd, head_size, bias=False)
        self.q    = nn.Linear(n_embd, head_size, bias=False)
        self.v    = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q    = self.k(x), self.q(x)
        wei     = q @ k.transpose(-2, -1) * C**-0.5
        wei     = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei     = F.softmax(wei, dim=-1)
        return self.drop(wei) @ self.v(x)

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size  = n_embd // n_heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj  = nn.Linear(n_embd, n_embd)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa  = MultiHeadAttention()
        self.ff  = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class mannLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[Block() for _ in range(n_layers)])
        self.ln      = nn.LayerNorm(n_embd)
        self.head    = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T    = idx.shape
        x       = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=device))
        x       = self.ln(self.blocks(x))
        logits  = self.head(x)
        loss    = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=5):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

# Initialize Model
model = mannLLM().to(device)

def train_model(cancel_flag=None, extra_text=""):
    print(f"mannLLM: Starting training on {device}...")

    # Merge base Q&A with any extra chat history
    combined = text
    if extra_text.strip():
        combined = text + ("\n\n" + extra_text.strip() + "\n\n") * 40
        print(f"mannLLM: Merged {len(extra_text)} chars of chat history into training data.")

    # Rebuild tokenizer and data from combined text
    chars_combined = sorted(set(combined))
    stoi_c = {c: i for i, c in enumerate(chars_combined)}
    itos_c = {i: c for i, c in enumerate(chars_combined)}
    encode_c = lambda s: [stoi_c[c] for c in s if c in stoi_c]
    decode_c = lambda l: ''.join([itos_c[i] for i in l])

    data_c     = torch.tensor(encode_c(combined), dtype=torch.long)
    n_c        = int(0.9 * len(data_c))
    train_c    = data_c[:n_c]
    val_c      = data_c[n_c:]

    def get_batch_c(split):
        d  = train_c if split == 'train' else val_c
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x  = torch.stack([d[i:i+block_size] for i in ix])
        y  = torch.stack([d[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(max_iters):
        if cancel_flag and cancel_flag.get("cancel"):
            print("mannLLM: Training cancelled.")
            return
        xb, yb       = get_batch_c('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % eval_every == 0 or step == max_iters - 1:
            model.eval()
            with torch.no_grad():
                xv, yv      = get_batch_c('val')
                _, val_loss = model(xv, yv)
            model.train()
            print(f"step {step:5d} | train loss {loss.item():.4f} | val loss {val_loss.item():.4f}")
    torch.save(model.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aman_model.pt'))
    print("mannLLM: Saved weights to aman_model.pt")

def generate_from_prompt(prompt_text, max_new_tokens=150):
    model.eval()
    formatted    = f"User: {prompt_text}\nBot:"
    clean_prompt = ''.join([c for c in formatted if c in stoi])

    if not clean_prompt:
        return "Error: Prompt contained no recognizable characters."

    prompt_ids = torch.tensor(encode(clean_prompt), dtype=torch.long).unsqueeze(0).to(device)
    output_ids = model.generate(prompt_ids, max_new_tokens=max_new_tokens)
    full_output = decode(output_ids[0].tolist())

    if "Bot:" in full_output:
        reply = full_output.split("Bot:")[-1].split("User:")[0].strip()
        return reply if reply else "I'm not sure. Try asking differently!"

    return full_output.strip()

# Only run training if executed directly (not when imported by the API)
if __name__ == '__main__':
    train_model()
    
    print("\n--- Interactive Test ---")
    while True:
        p = input("Prompt (or 'quit'): ")
        if p == 'quit': break
        print(generate_from_prompt(p))
        print("-" * 20)