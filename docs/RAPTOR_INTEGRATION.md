# RAPTOR Integration Guide

This document explains the RAPTOR integration and how to evaluate its performance against standard RAG approaches.

## What is RAPTOR?

**R**ecursive **A**bstractive **P**rocessing for **T**ree-**O**rganized **R**etrieval

RAPTOR builds a hierarchical tree from document chunks:

```
Level 3 (Root):     [Summary of all themes]
                    /         |         \
Level 2:    [Theme A]     [Theme B]     [Theme C]
            /       \      /       \      /      \
Level 1:  [Sub1]  [Sub2] [Sub3] [Sub4] [Sub5] [Sub6]
          /    \   /    \  ...
Level 0: [Original chunks with full detail]
```

## Key Benefits

| Benefit | Description |
|---------|-------------|
| **Multi-scale Retrieval** | Retrieve high-level summaries OR specific details |
| **Better Thematic Questions** | "What are the main themes?" works better |
| **Context Compression** | Summaries reduce token usage vs full chunks |
| **Hierarchical Organization** | Natural document structure preserved |

## Integration Overview

### New Files Added

```
src/
├── raptor/
│   ├── __init__.py
│   ├── models.py           # TreeNode, RaptorTree
│   ├── clustering.py       # GMM + UMAP clustering
│   ├── tree_builder.py     # Build tree from chunks
│   └── retriever.py        # Retrieve from tree
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py          # Recall, Precision, NDCG, MRR
│   └── runner.py           # Evaluation framework
├── pipeline_raptor.py      # Extended pipeline with RAPTOR
└── ingestion_chroma.py     # Chroma alternative to FAISS

scripts/
└── compare_retrieval.py    # Run comparisons
```

## Usage

### 1. Build RAPTOR Trees

```python
from src.pipeline_raptor import RaptorPipeline, raptor_configs

# Initialize with RAPTOR enabled
pipeline = RaptorPipeline('.', raptor_configs["raptor_basic"])

# Process documents (builds trees automatically)
pipeline.parse_pdf_reports()
pipeline.process_parsed_reports()  # This now builds RAPTOR trees too

# Check tree stats
print(pipeline.get_raptor_stats())
```

### 2. Query with RAPTOR

```python
# Query uses RAPTOR automatically
answer = pipeline.query_single(
    "What are the company's main growth strategies?",
    use_raptor=True  # or False for standard retrieval
)

# Answer includes RAPTOR metadata:
# {
#   "final_answer": "...",
#   "retrieval_method": "raptor",
#   "retrieval_strategy": "multi_level",
#   "tree_levels_accessed": [0, 1, 2],
#   "retrieved_nodes": [...]
# }
```

### 3. Run Evaluation

```bash
# Create test queries
cat > test_queries.json << 'EOF'
[
  {
    "question": "What was the revenue in 2023?",
    "answer": "$100 million",
    "relevant_chunks": ["5", "12"],
    "difficulty": "easy"
  },
  {
    "question": "What are the main risks?",
    "answer": "Market volatility, regulatory changes",
    "relevant_chunks": ["20", "21", "22"],
    "difficulty": "medium"
  }
]
EOF

# Run comparison
python scripts/compare_retrieval.py \
    --root-path . \
    --queries test_queries.json \
    --output-dir evaluation_results
```

## Evaluation Metrics Explained

### Ranking Metrics

| Metric | What it Measures | Good For |
|--------|------------------|----------|
| **Recall@K** | % of relevant chunks found | Completeness |
| **Precision@K** | % of retrieved chunks that are relevant | Quality |
| **F1@K** | Balance of Recall and Precision | Overall |
| **NDCG@K** | Ranking quality (earlier = better) | Order matters |
| **MRR** | Position of first relevant result | Quick answers |

### Efficiency Metrics

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| **Latency** | Time to retrieve | User experience |
| **Tokens Used** | LLM tokens for summary | Cost |
| **Index Size** | Storage requirements | Scalability |

## Expected Performance Comparison

### For Thematic Questions (e.g., "What are the main themes?")

| Strategy | Recall@5 | NDCG@5 | Notes |
|----------|----------|--------|-------|
| FAISS Flat | 0.60 | 0.55 | May miss high-level patterns |
| Chroma | 0.62 | 0.57 | Similar to FAISS |
| RAPTOR Multi-Level | **0.85** | **0.80** | ✅ Accesses summary nodes |
| RAPTOR Leaf-Only | 0.60 | 0.55 | Same as flat |

### For Specific Facts (e.g., "What was revenue in Q3?")

| Strategy | Recall@5 | Latency | Notes |
|----------|----------|---------|-------|
| FAISS Flat | 0.90 | 15ms | ✅ Fast and accurate |
| Chroma | 0.90 | 20ms | Similar performance |
| RAPTOR Multi-Level | 0.88 | 45ms | Slightly slower |
| RAPTOR Leaf-Only | 0.90 | 30ms | Same accuracy, slower |

## When to Use Each Strategy

### Use **FAISS Flat** when:
- ✅ Documents are short (< 50 pages)
- ✅ Questions are factual/specific
- ✅ Latency is critical
- ✅ Simple setup preferred

### Use **Chroma** when:
- ✅ Need metadata filtering (date, author, etc.)
- ✅ Want easy document updates
- ✅ Prefer Python-native solution

### Use **RAPTOR** when:
- ✅ Documents are long (100+ pages)
- ✅ Thematic/synthesis questions common
- ✅ Multi-document comparison needed
- ✅ Scale to 1000+ documents planned

## Configuration Options

### RAPTOR Configs

```python
from src.pipeline_raptor import raptor_configs

# Basic: 2 levels, fast
raptor_configs["raptor_basic"]

# Deep: 4 levels, more abstraction
raptor_configs["raptor_deep"]

# Leaf-only: Uses RAPTOR structure but only retrieves leaves
raptor_configs["raptor_leaf_only"]

# Standard: Disable RAPTOR, use flat retrieval
raptor_configs["standard"]
```

### Custom Config

```python
from src.pipeline_raptor import RaptorRunConfig

config = RaptorRunConfig(
    use_raptor=True,
    raptor_max_levels=3,           # Tree depth
    raptor_cluster_threshold=0.5,  # GMM probability threshold
    raptor_top_k_per_level=2,      # Nodes per level
    raptor_strategy="multi_level"  # "multi_level", "leaf_only", "root_to_leaf"
)
```

## Interpreting Results

### Example Output

```
BEST STRATEGY BY METRIC:
  recall@5             raptor_multi_level        (0.850)
  precision@5          faiss_flat                (0.720)
  f1@5                 raptor_multi_level        (0.780)
  ndcg@5               raptor_multi_level        (0.800)
  mrr                  faiss_flat                (0.650)
  latency_ms           faiss_flat                (15.2)

DETAILED SCORES:
  Strategy                Recall@5   Prec@5     F1@5       NDCG@5     MRR
  --------------------------------------------------------------------------------
  faiss_flat              0.600      0.720      0.655      0.550      0.650
  chroma                  0.620      0.710      0.660      0.570      0.640
  raptor_multi_level      0.850      0.680      0.780      0.800      0.580
  raptor_leaf_only        0.600      0.700      0.645      0.550      0.630
```

### What This Tells You

1. **RAPTOR Multi-Level** has best recall and NDCG → Better at finding relevant content
2. **FAISS Flat** has lowest latency → Best for speed-critical applications
3. **RAPTOR** is best for thematic questions, **FAISS** for specific facts

## Next Steps

1. **Run baseline evaluation** with your 20 PDFs
2. **Identify question types** (thematic vs specific)
3. **Choose strategy** based on your use case
4. **Fine-tune parameters** (levels, thresholds)
5. **Scale up** as you add more documents
