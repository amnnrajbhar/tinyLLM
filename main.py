# ============================================================
# TINY GPT — TRAINED ON AMAN'S PERSONAL INFO
# A character-level Transformer built from scratch
# ============================================================

import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================
# SECTION 1: THE DATASET (Extracted from Portfolio HTML)
# ============================================================

text = """
Aman Rajbhar is a Software Engineer based in Mumbai, Maharashtra, India.
Aman is an expert in Angular and ASP.NET web development.
Aman currently works at Clover Infotech Private Limited as a Software Engineer since December 2024.
At Clover Infotech, Aman develops high-performance web applications using Angular, TypeScript, and RxJS.
Aman designs and implements scalable microservices and REST APIs.
Aman previously worked at Benchmark Computer Solutions Limited as a Junior Software Developer from August 2022 to December 2024.
At Benchmark, Aman built dynamic web apps using Angular, ASP.NET, and MVC.
Aman graduated from the University of Mumbai with a B.Sc. in Computer Science in May 2022.
Aman is currently pursuing an MCA in Computer Applications from IGNOU since June 2022.
Aman is skilled in frameworks including Angular, ASP.NET MVC, .NET Core, React, and Node.js.
Aman is skilled in programming languages including JavaScript, TypeScript, C#, Python, HTML, CSS, and SQL.
Aman is experienced with databases including SQL Server, Firebase Realtime Database, MySQL, and MongoDB.
Aman utilizes tools and platforms like Visual Studio, Git, GitHub, XAMPP, and Crystal Reports.
Aman applies Agile methodologies, Scrum, and CI/CD practices in his workflow.
Aman built a GenAI Web App, which is an AI-powered chatbot using Ionic, Angular, and Google Generative AI.
Aman built a Media Upload Web App utilizing AWS S3 Buckets and Angular.
Aman built an HRMS Portal, a Human Resource Management System using Angular, Node.js, and Tailwind CSS.
Aman built an LMS Portal, a Learning Management System using Angular and Firebase.
Aman built a Digital Loan Process App using Angular, Node.js, and MongoDB.
Aman built a Property Seller platform for property listing.
Aman built a TV Services provider website using HTML, CSS, and JavaScript.
Aman built a Weather Web App for real-time weather updates using OpenWeather API.
Aman built a personal Portfolio Website using Angular, TypeScript, SCSS, and Bootstrap.
Aman holds a certification in Advanced Angular from Great Learning.
Aman holds a certification in ASP.NET MVC Development.
Aman can be contacted via email at amanrajbhar289@gmail.com.
Aman optimizes application performance, successfully reducing load times by 70 percent.
Aman manages version control across multiple projects using Git and GitHub.
"""

text = text.strip()
text = (text + "\n") * 100  # Repeat to build volume

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

class TinyGPT(nn.Module):
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
model = TinyGPT().to(device)

def train_model():
    print(f"Starting training on {device}...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(max_iters):
        xb, yb       = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % eval_every == 0 or step == max_iters - 1:
            model.eval()
            with torch.no_grad():
                xv, yv      = get_batch('val')
                _, val_loss = model(xv, yv)
            model.train()
            print(f"step {step:5d} | train loss {loss.item():.4f} | val loss {val_loss.item():.4f}")
    
    # Save the trained weights so the API can load them without retraining
    torch.save(model.state_dict(), 'aman_model.pt')
    print("Model saved to aman_model.pt")

def generate_from_prompt(prompt_text, max_new_tokens=150):
    model.eval()
    # Filter out unknown characters to prevent crashes
    clean_prompt = ''.join([c for c in prompt_text if c in stoi])
    if not clean_prompt:
        return "Error: Prompt contained no recognizable characters."
        
    prompt_ids = torch.tensor(encode(clean_prompt), dtype=torch.long).unsqueeze(0).to(device)
    output_ids = model.generate(prompt_ids, max_new_tokens=max_new_tokens)
    return decode(output_ids[0].tolist())

# Only run training if executed directly (not when imported by the API)
if __name__ == '__main__':
    train_model()
    
    print("\n--- Interactive Test ---")
    while True:
        p = input("Prompt (or 'quit'): ")
        if p == 'quit': break
        print(generate_from_prompt(p))
        print("-" * 20)