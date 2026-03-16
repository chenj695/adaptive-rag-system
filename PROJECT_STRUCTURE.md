# Atmospheric Science RAG System - Project Structure

## Overview

A comprehensive RAG (Retrieval-Augmented Generation) system designed for the Atmospheric Science Knowledge Competition. Features multiple retrieval backends (FAISS, Chroma, RAPTOR) with a beautiful weather-themed web interface.

## Complete File Structure

```
rag_system/
│
├── .env                              # Environment variables (API keys)
├── .gitignore                        # Git ignore rules
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
├── setup.py                          # Setup script
├── main.py                           # CLI entry point
├── PROJECT_STRUCTURE.md              # This file
│
├── docs/                             # Documentation
│   ├── RAPTOR_INTEGRATION.md         # RAPTOR integration guide
│   └── vector_db_comparison.md       # FAISS vs Chroma comparison
│
├── scripts/                          # Utility scripts
│   └── compare_retrieval.py          # Retrieval strategy comparison tool
│
├── src/                              # Core source code
│   ├── __init__.py
│   │
│   ├── pipeline.py                   # Main FAISS pipeline
│   ├── pipeline_raptor.py            # Extended RAPTOR pipeline
│   │
│   ├── pdf_parsing.py                # PDF extraction with Docling
│   ├── parsed_reports_merging.py     # Convert parsed to markdown
│   ├── text_splitter.py              # Text chunking
│   │
│   ├── ingestion.py                  # FAISS vector DB creation
│   ├── ingestion_chroma.py           # Chroma vector DB creation
│   │
│   ├── retrieval.py                  # FAISS retrieval
│   ├── reranking.py                  # LLM-based reranking
│   │
│   ├── prompts.py                    # LLM prompts and schemas
│   ├── questions_processing.py       # Q&A logic
│   │
│   ├── tables_serialization.py       # Table processing
│   │
│   ├── evaluation/                   # Evaluation framework
│   │   ├── __init__.py
│   │   ├── metrics.py                # Recall, Precision, NDCG, MRR
│   │   └── runner.py                 # Evaluation runner
│   │
│   └── raptor/                       # RAPTOR implementation
│       ├── __init__.py
│       ├── models.py                 # TreeNode, RaptorTree
│       ├── clustering.py             # GMM + UMAP clustering
│       ├── tree_builder.py           # Build hierarchical trees
│       └── retriever.py              # Multi-level retrieval
│
└── web/                              # Web interface
    ├── __init__.py
    ├── app.py                        # Flask app (FAISS)
    ├── app_chroma.py                 # Flask app (Chroma)
    ├──
    ├── templates/
    │   └── index.html                # Weather-themed UI
    │
    └── static/
        ├── style.css                 # Animated sky, clouds, weather symbols
        └── app.js                    # Frontend logic

## Data Directories (Created at Runtime)

```
data/
├── pdf_reports/          # Upload your atmospheric science PDFs here
├── uploads/              # Temporary upload storage
├── debug/                # Intermediate processing data
│   ├── data_01_parsed_reports/
│   ├── data_02_merged_reports/
│   └── data_03_reports_markdown/
├── databases/            # Vector databases
│   ├── chunked_reports/
│   └── vector_dbs/       # .faiss files
└── chroma_db/            # Chroma persistent storage
└── raptor_trees/         # RAPTOR tree files
```

## Module Descriptions

### Core Pipeline (`src/pipeline.py`)
- Main orchestration for document processing
- Supports FAISS vector search
- CLI interface with Click

### RAPTOR Pipeline (`src/pipeline_raptor.py`)
- Extended pipeline with hierarchical tree building
- Multi-level retrieval strategies
- Configurable tree depth and clustering

### PDF Processing (`src/pdf_parsing.py`)
- Uses Docling for PDF extraction
- Extracts text, tables, and structure
- Supports parallel processing

### Vector Databases

#### FAISS (`src/ingestion.py`, `src/retrieval.py`)
- Facebook AI Similarity Search
- Fast cosine similarity search
- File-based storage (.faiss)

#### Chroma (`src/ingestion_chroma.py`)
- Native metadata filtering
- Persistent storage
- Python-native API

### RAPTOR (`src/raptor/`)
- **tree_builder.py**: Builds hierarchical trees from chunks
- **clustering.py**: GMM + UMAP soft clustering
- **retriever.py**: Multi-level tree traversal
- **models.py**: Tree data structures

### Evaluation (`src/evaluation/`)
- **metrics.py**: Recall@K, Precision@K, NDCG@K, MRR, F1
- **runner.py**: Automated evaluation framework
- Statistical significance testing

### Web Interface (`web/`)
- Flask backend with CORS
- Weather-themed CSS animations
- Real-time upload and processing

## Key Features

1. **Multiple Backends**: FAISS, Chroma, RAPTOR
2. **Beautiful UI**: Animated sky, clouds, weather symbols
3. **Evaluation Framework**: Comprehensive metrics for comparison
4. **Modular Design**: Easy to extend and customize
5. **Atmospheric Focus**: Designed for weather/climate documents

## Usage Flow

```
1. Upload PDFs (atmospheric science papers)
   ↓
2. Process Documents (parse → chunk → index)
   ↓
3. Query (natural language questions)
   ↓
4. Evaluate (metrics and comparisons)
```

## Dependencies

See `requirements.txt` for complete list. Key dependencies:
- Flask (web framework)
- OpenAI (embeddings and LLM)
- FAISS/Chroma (vector search)
- Docling (PDF parsing)
- scikit-learn (clustering for RAPTOR)

## License

MIT License - Based on RAG Challenge 2 winning solution
