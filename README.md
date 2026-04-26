# 💰 FinanceGPT — India's Multilingual Finance Assistant

**Ask anything about Indian finance in any language — Hindi, Telugu, Tamil, or English.**

> Built with ❤️ at ₹0 cost | Powered by Groq + Qdrant + Streamlit

---

## 🌟 What It Does

FinanceGPT answers questions about:
- 📈 Mutual Funds (SIP, NAV, ELSS, AMC comparison)
- 💹 Stock Market (NSE, BSE, Nifty, IPOs, fundamentals)
- 🧾 Taxation (income tax slabs, 80C, LTCG, ITR filing)
- 🏦 Banking (FDs, RBI policy, bonds, T-bills)
- 🏠 Loans & Credit (home loans, EMI, CIBIL score, Mudra)
- 📊 Financial Planning (retirement, NPS, PPF, insurance)
- 🪙 Crypto (30% tax, regulations, VDA)

---

## 🏗️ Tech Stack (All Free)

| Component       | Technology            | Free Tier              |
|-----------------|-----------------------|------------------------|
| Frontend        | Streamlit             | HuggingFace Spaces     |
| Backend API     | FastAPI               | Render (750 hrs/month) |
| Vector DB       | Qdrant Cloud          | 1 GB free              |
| LLM             | Groq (Llama 3.3 70B)  | 14,400 req/day         |
| Embeddings      | sentence-transformers | Runs locally, free     |
| Memory          | SQLite                | Free, unlimited        |
| CI/CD           | GitHub Actions        | Free                   |
| Uptime          | UptimeRobot           | Free (50 monitors)     |

**Total Monthly Cost: ₹0**

---

## 🚀 Step-by-Step Setup Guide (For Complete Beginners)

### Step 1: Get All Free Accounts

1. **GitHub** — https://github.com → Sign up free
2. **Groq API** — https://console.groq.com → Sign up → Create API Key
3. **Qdrant Cloud** — https://cloud.qdrant.io → Sign up → Create cluster → Get URL + API key
4. **HuggingFace** — https://huggingface.co → Sign up → Create Space (Streamlit)
5. **Render** — https://render.com → Sign up with GitHub
6. **UptimeRobot** — https://uptimerobot.com → Sign up free

### Step 2: Clone and Set Up Project

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/financegpt.git
cd financegpt

# Install Python (if not installed)
# Download from: https://python.org (version 3.10 or higher)

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
```

### Step 3: Fill in .env File

Open `.env` in any text editor (Notepad, VS Code) and fill in:
```
GROQ_API_KEY=gsk_your_actual_key_here
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_key_here
BACKEND_URL=http://localhost:8000
```

### Step 4: Load Data into Qdrant (Run Once)

```bash
python data/ingest.py
```
This will:
- Download the multilingual embedding model (~120 MB, one-time)
- Process all finance documents
- Upload them to your Qdrant Cloud cluster

You should see: `✅ Done! 40 documents uploaded to Qdrant.`

### Step 5: Run Locally (Test It)

Open TWO terminal windows:

**Terminal 1 — Backend:**
```bash
uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
streamlit run frontend/app.py
```

Open your browser at `http://localhost:8501` — FinanceGPT is running!

Test it by asking: *"What is SIP?"* or *"SIP ante enti?"*

---

## ☁️ Deploy to Production (Free)

### Deploy Backend to Render

1. Push your code to GitHub
2. Go to render.com → New → Web Service
3. Connect your GitHub repo
4. Render auto-reads `render.yaml`
5. Go to Environment → Add these variables:
   - `GROQ_API_KEY` = your Groq key
   - `QDRANT_URL` = your Qdrant URL
   - `QDRANT_API_KEY` = your Qdrant key
6. Deploy! You'll get a URL like `https://financegpt-api.onrender.com`

### Keep Render Awake (UptimeRobot)

Render's free tier sleeps after 15 minutes. Fix this:
1. Go to uptimerobot.com → Add New Monitor
2. Monitor type: HTTP
3. URL: `https://your-app.onrender.com/health`
4. Monitoring interval: 10 minutes
5. Done! Server stays awake 24/7.

### Deploy Frontend to HuggingFace Spaces

1. Go to huggingface.co → New Space
2. Space name: `financegpt`
3. SDK: Streamlit
4. Upload `frontend/app.py` and `requirements.txt`
5. Add Secret: `BACKEND_URL` = your Render URL
6. Space builds automatically — you get a public URL!

### Set Up CI/CD (Auto-Deploy on Code Push)

In GitHub repo → Settings → Secrets → Add:
- `RENDER_DEPLOY_HOOK_URL` (from Render → your service → Deploy Hook)
- `HF_TOKEN` (from HuggingFace → Settings → Access Tokens)
- `HF_USERNAME` (your HuggingFace username)
- `HF_SPACE_NAME` = `financegpt`

Now every `git push` auto-deploys both frontend and backend!

---

## 📁 Project Structure

```
financegpt/
├── backend/
│   ├── main.py        ← FastAPI app (endpoints: /chat, /health, /clear)
│   ├── rag.py         ← RAG pipeline (retrieve + generate)
│   └── memory.py      ← SQLite conversation memory
├── frontend/
│   └── app.py         ← Streamlit chat UI
├── data/
│   ├── finance_data.json  ← 40+ finance knowledge chunks
│   └── ingest.py      ← Load data into Qdrant (run once)
├── .github/
│   └── workflows/
│       └── deploy.yml ← Auto-deploy CI/CD
├── .env.example       ← Environment variable template
├── render.yaml        ← Render deployment config
├── requirements.txt   ← Python dependencies
└── README.md          ← This file
```

---

## 🔒 Legal Safety

Every response ends with a SEBI disclaimer. FinanceGPT:
- ✅ Provides financial education
- ✅ Explains concepts, rules, and calculations
- ❌ Does NOT give stock tips or "buy/sell" signals
- ❌ Does NOT provide F&O or derivatives advice
- ❌ Is NOT a SEBI-registered investment adviser

---

## 🛠️ Adding More Finance Data

To add more knowledge to FinanceGPT, edit `data/finance_data.json`:

```json
{
  "id": "your_unique_id",
  "domain": "personal_finance",
  "language": "en",
  "title": "Your Topic Title",
  "content": "Detailed explanation here...",
  "keywords": ["keyword1", "keyword2"]
}
```

Available domains: `personal_finance`, `mutual_funds`, `stock_market`, 
`taxation`, `banking`, `crypto`, `regulation`, `macroeconomics`, 
`loans_credit`, `financial_planning`

After adding, run `python data/ingest.py` again.

---

## 🤝 Contributing

Found a mistake? Want to add more multilingual content?
1. Fork the repo
2. Create a branch: `git checkout -b add-marathi-content`
3. Make changes
4. Submit a Pull Request

---

*Built for every Indian who deserves quality financial education regardless of language or income.*
