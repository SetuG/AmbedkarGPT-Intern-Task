Q1) The script is designed to pick the best configuration automatically by a weighted overall score in most similar RAG setups the “medium” chunk size (here 550 chars with overlap 55) gives the best balance between retrieval recall and answer quality.

Q2)Large chunks work best overall because they provide the highest answer quality and are recommended for production use.

Q3)Retrieval misses incomplete answers and model hallucinations.

q4)We can add more context in prompts increase retrieval depth and apply reranking to improve chunk relevance.
