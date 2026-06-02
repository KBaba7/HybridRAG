# HybridRAG — 10-K Financial Analyst

Upload a 10-K filing and ask questions about it. The app parses the PDF, embeds the content into a vector database, and uses Groq to answer questions with the actual document as context.

## What it does

- Parses PDFs with spatial awareness (so tables stay intact)
- Chunks and embeds text into Qdrant for semantic search
- Uses Groq to answer financial questions, extract tables, and generate charts
- Remembers your documents between sessions — no re-uploading needed

## Stack

- **Backend** — FastAPI, LiteParse, SentenceTransformers, Qdrant, Groq
- **Frontend** — Vanilla JS, Chart.js

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/KBaba7/HybridRAG.git
cd HybridRAG
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Configure environment**

```bash
cp .env.example backend/.env
```

Fill in your keys in `backend/.env`:
- `GROQ_API_KEY` — from [console.groq.com](https://console.groq.com)
- `QDRANT_URL` + `QDRANT_API_KEY` — from [cloud.qdrant.io](https://cloud.qdrant.io)

**3. Run the backend**

```bash
cd backend
uvicorn main:app --reload
```

**4. Open the frontend**

Just open `frontend/index.html` in your browser.

## Requirements

- Python 3.10+
- [Tesseract](https://brew.sh) for OCR: `brew install tesseract`
- [Poppler](https://brew.sh) for PDF rendering: `brew install poppler`
