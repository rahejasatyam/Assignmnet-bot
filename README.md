# SHL Assessment Recommender

A production-ready Conversational AI that helps hiring managers find the right SHL assessments through natural dialogue.

## Architecture

```
POST /chat (full conversation history)
      │
      ▼
  agent.py  ← builds composite query from all user messages
      │
      ├─► retriever.py  ← FAISS semantic search → top-15 catalog items
      │
      ├─► system_prompt.py  ← injects catalog context + turn number
      │
      ├─► Groq LLM (llama-3.3-70b-versatile)  ← generates JSON response
      │
      └─► URL validation  ← strips any hallucinated URLs
      │
      ▼
  ChatResponse  ← {reply, recommendations, end_of_conversation}
```

## Setup (Local)

### 1. Clone and install
```bash
pip install -r requirements.txt
```

### 2. Set your Groq API key
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
# Free key: https://console.groq.com/
```

### 3. Scrape the catalog (already done — data/catalog.json included)
```bash
python scraper/scrape_catalog.py
```

### 4. Build the FAISS index
```bash
python retriever/build_index.py
```

### 5. Run the server
```bash
uvicorn api.main:app --reload --port 8000
```

### 6. Test it
```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I need tests for a Java developer"}]}'
```

## API Reference

### GET /health
```json
{"status": "ok"}
```

### POST /chat
**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."}
  ]
}
```

**Response (always):**
```json
{
  "reply": "Here are the best assessments for a mid-level Java developer...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/products/product-catalog/view/java-8-new/", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/products/product-catalog/view/opq32r/", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

## Running Tests
```bash
pytest tests/test_agent.py -v
```

## Deployment (Render)
1. Push this repo to GitHub
2. Create a new Web Service on [Render](https://render.com)
3. Connect your GitHub repo
4. Set `GROQ_API_KEY` in the Render environment variables panel
5. Render will use `render.yaml` for build and start commands automatically

## Project Structure
```
assignment/
├── scraper/
│   └── scrape_catalog.py      # Scrapes SHL catalog → data/catalog.json
├── retriever/
│   ├── retriever.py           # FAISS semantic search over catalog
│   └── build_index.py         # Standalone index builder script
├── agent/
│   ├── system_prompt.py       # System prompt with dynamic catalog context
│   └── agent.py               # Core agent pipeline
├── api/
│   └── main.py                # FastAPI app with /health and /chat
├── tests/
│   └── test_agent.py          # 15 test scenarios
├── data/
│   └── catalog.json           # Scraped SHL catalog (ground truth)
├── index/                     # FAISS index files (built from catalog)
├── requirements.txt
├── render.yaml
└── .env.example
```

## Environment Variables
| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ Yes | Free key from https://console.groq.com/ |
| `PORT` | No | Server port (default: 8000, auto-set by Render) |
