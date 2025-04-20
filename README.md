# 🧠 Validated Pattern Assistant UI

This project provides a lightweight Gradio-based web chat interface for interacting with a Retrieval-Augmented Generation (RAG) API server backed by Qdrant and multiple LLMs.

It is designed for exploring and answering questions related to **Validated Patterns** using either local or remote models (e.g., vLLM, OpenAI).

---

## ✨ Features

- 🔍 Query documentation using semantic search + LLM
- 🔀 Choose from multiple LLMs (Mistral, GPT-4, etc.)
- 🧠 Retains conversation history
- 📡 Real-time streaming responses
- 📦 Simple containerized deployment

---

## 🚀 Running Locally

### 1. Set API endpoint

```bash
export RAG_API_URL=http://localhost:8080
```

### 2. Run the UI

```bash
python app.py
```

Then open [http://localhost:7860](http://localhost:7860) in your browser.

---

## 🔧 Configuration

The UI queries `GET /models` on the backend to dynamically list available models, based on your backend's `.env` or deployment configuration.

Each user query sends:
- The selected model name
- The current user input
- The chat history so far

---

## 📚 Backend

This UI is designed to work with [vp-rag-llm-api](https://github.com/dminnear-rh/vp-rag-llm-api), a FastAPI app that performs:
- Embedding search with Qdrant
- Semantic re-ranking
- LLM completions via vLLM or OpenAI
