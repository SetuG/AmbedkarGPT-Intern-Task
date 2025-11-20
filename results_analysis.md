# RAG Evaluation Results - Detailed Analysis

**Evaluation Date:** 2025-11-20 17:22:24

---

## Executive Summary

**Best Configuration:** large_chunks
- Chunk Size: 900
- Chunk Overlap: 90
- Overall Score: 0.530

## Detailed Metrics Comparison

| Metric | Small Chunks | Medium Chunks | Large Chunks |
|--------|--------------|---------------|---------------|
| Hit Rate | 0.800 | 0.880 | 0.880 |
| MRR | 0.753 | 0.807 | 0.820 |
| Precision@K | 0.600 | 0.547 | 0.427 |
| ROUGE-L | 0.292 | 0.289 | 0.316 |
| Cosine Similarity | 0.529 | 0.571 | 0.555 |
| Faithfulness | 0.434 | 0.494 | 0.584 |

## Recommendations

### 1. Optimal Configuration

Use **large_chunks** for production:
- Provides best balance of retrieval accuracy and answer quality
- Chunk size: 900 characters
- Chunk overlap: 90 characters

### 2. Areas for Improvement

- **Faithfulness**: Implement answer verification and re-ranking

---

*Report generated automatically by analyze_results.py*
