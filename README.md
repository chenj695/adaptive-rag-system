# RAG for Meteorology Knowledge Contest

A complete RAG (Retrieval-Augmented Generation) system with hierarchical retrieval (RAPTOR), multiple backends (FAISS/Chroma), and comprehensive evaluation framework. Built for meteorology knowledge contests with a beautiful weather-themed UI featuring animated sky, clouds, and sun.


## Demo

The system features a clean atmospheric science-themed interface:
- Animated gradient sky background
- Floating clouds with drift animation
- Pulsing sun with rotating rays
- Drag-and-drop PDF upload
- Natural language query interface

<p align="center">
  <img src="demo.png" width="66%" alt="Meteorology Knowledge Contest UI Demo">
</p>

## Features

- 📄 **PDF Processing**: Extract text, tables, and structure from PDFs using Docling
- 🔍 **Multi-Path Retrieval**: Semantic search (FAISS) + Lexical search(BM25) and RRF Fusion (Reciprocal Rank Fusion)
- 🧠 **LLM Reranking**: GPT-4o-mini reranks retrieved chunks for better accuracy
- 🌐 **Web Interface**: Modern React-style UI with animated weather theme (sky, clouds, sun)
- ⚡ **Parallel Processing**: Multi-process PDF parsing for speed
- 🌳 **RAPTOR Support**: Hierarchical retrieval with tree-based document organization
- 📊 **Evaluation Framework**: Comprehensive metrics (Recall, Precision, NDCG, MRR)
- 🔒 **Fully Offline**: Local embeddings with sentence-transformers (no API calls for vectorization)

## Project Structure

```
rag_system/
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys)
├── README.md
├── data/
│   ├── pdf_reports/       # Place your PDFs here
│   ├── debug/             # Intermediate processing data
│   ├── databases/         # Vector databases
│   └── uploads/           # Web UI uploads
├── src/
│   ├── __init__.py
│   ├── pipeline.py        # Main orchestration
│   ├── prompts.py         # LLM prompts & schemas
│   ├── pdf_parsing.py     # PDF extraction with Docling
│   ├── parsed_reports_merging.py  # Convert to markdown
│   ├── text_splitter.py   # Chunk documents
│   ├── ingestion.py       # Create vector DBs
│   ├── retrieval.py       # Vector search
│   ├── reranking.py       # LLM reranking
│   ├── tables_serialization.py  # Table processing
│   └── questions_processing.py  # Q&A logic
└── web/
    ├── __init__.py
    ├── app.py             # Flask backend
    ├── templates/
    │   └── index.html     # Main UI
    └── static/
        ├── style.css      # Modern styling
        └── app.js         # Frontend logic
```

## Setup

1. **Install Dependencies**
```bash
cd rag_system
pip install -r requirements.txt
```

2. **Set OpenAI API Key**
```bash
# Edit .env file
OPENAI_API_KEY=your_key_here
```

3. **Add Your PDFs**
Copy your PDF files to `data/pdf_reports/`

## Usage

### Option 1: Web UI (Recommended)

```bash
python main.py webui
```

Then open http://localhost:5000 in your browser.

Features:
- 📤 Drag & drop PDF upload
- ⚙️ One-click document processing
- ❓ Natural language questions
- 📊 View reasoning and retrieved context

### Option 2: Command Line

```bash
# Process all PDFs (parse, chunk, create vector DBs)
python main.py parse-pdfs --parallel
python main.py process-reports

# Or do it all at once via Python
python -c "from src.pipeline import Pipeline; p = Pipeline('.'); p.parse_pdf_reports(); p.process_parsed_reports()"
```

### Query via Python

```python
from src.pipeline import Pipeline

pipeline = Pipeline('.')
answer = pipeline.query_single("What was the total revenue in 2023?", document_name="annual_report")

print(answer['final_answer'])
print(answer['reasoning_summary'])
```

## How It Works

1. **PDF Parsing**: Docling extracts text, tables, and structure
2. **Text Splitting**: Documents chunked with 300-token overlap
3. **Vectorization**: Local `sentence-transformers` model creates embeddings (offline, no API)
4. **Retrieval**: FAISS finds top-k similar chunks
5. **Reranking**: GPT-4o-mini scores relevance 0-1
6. **Answer Generation**: o3-mini generates structured answer with reasoning

## Configuration

Edit `src/pipeline.py` to customize:

- `chunk_size`: Text splitter chunk size (default: 300)
- `top_n_retrieval`: Number of chunks to retrieve (default: 6)
- `llm_reranking`: Enable/disable reranking
- `answering_model`: Change LLM model

### Using GitHub Models (Free Tier Available)

The system now supports **GitHub Models API** as an alternative to OpenAI:

1. **Get a GitHub Token**: Create a Personal Access Token at https://github.com/settings/tokens with `models:read` scope

2. **Configure Environment**:
```bash
# Edit .env file
GITHUB_TOKEN=ghp_your_token_here
# Optional: choose model (default: openai/gpt-4.1)
GITHUB_MODEL=openai/gpt-4.1
```

3. **Use GitHub Models in Pipeline**:
```python
from src.pipeline import Pipeline, configs

# Use GitHub Models
pipeline = Pipeline('.', run_config=configs['github'])

# Or use GitHub Models with multi-path retrieval
pipeline = Pipeline('.', run_config=configs['github_multi'])
```

**Available GitHub Models**:
- `openai/gpt-4.1` - Latest GPT-4.1 (recommended)
- `openai/gpt-4o` - Vision-capable
- `openai/gpt-4o-mini` - Fast and cheap
- `anthropic/claude-3.5-sonnet` - Great reasoning
- `meta/llama-3.3-70b-instruct` - Open source
- `deepseek/deepseek-r1` - Reasoning model

The WebUI automatically uses the configured provider based on your pipeline settings.

## Requirements

- Python 3.9+
- OpenAI API key (only for LLM generation, not embeddings)
- 8GB+ RAM (for PDF processing)
- Optional: GPU for faster Docling processing

## Embedding Model

The system uses **local sentence-transformers** for embeddings:
- **Default**: `all-MiniLM-L6-v2` (384 dimensions, fast)
- **Alternative**: `all-mpnet-base-v2` (better quality, slower)
- First run downloads ~80MB model to local cache
- No API calls required - works completely offline for vectorization

To use a different model, set in `.env`:
```bash
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## License

MIT License - Based on [RAG Challenge 2](https://github.com/IlyaRice/RAG-Challenge-2)
