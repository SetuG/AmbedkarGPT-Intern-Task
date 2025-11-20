"""
Comprehensive RAG Evaluation Framework
Evaluates retrieval quality, answer quality, and semantic similarity
"""

import json
import os
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import time

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# Evaluation metrics imports
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class RAGEvaluator:
    """Comprehensive RAG evaluation system"""
    
    def __init__(self, corpus_path="./corpus", persist_dir="./eval_chroma_db"):
        self.corpus_path = corpus_path
        self.persist_dir = persist_dir
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.results = {
            "retrieval_metrics": {},
            "answer_quality_metrics": {},
            "semantic_metrics": {},
            "per_question_results": []
        }
        
    def load_corpus(self) -> List:
        """Load all documents from corpus folder (PDF and TXT)"""
        print(f"\n{'='*60}")
        print("Loading Document Corpus")
        print(f"{'='*60}")
        
        documents = []
        files_loaded = 0
        
        for filename in os.listdir(self.corpus_path):
            filepath = os.path.join(self.corpus_path, filename)
            
            try:
                if filename.endswith('.txt'):
                    loader = TextLoader(filepath, encoding='utf-8')
                    docs = loader.load()
                    documents.extend(docs)
                    files_loaded += 1
                    print(f"✓ Loaded: {filename} ({len(docs[0].page_content)} chars)")
                    
                elif filename.endswith('.pdf'):
                    loader = PyPDFLoader(filepath)
                    docs = loader.load()
                    documents.extend(docs)
                    files_loaded += 1
                    print(f"✓ Loaded: {filename} ({len(docs)} pages)")
                    
            except Exception as e:
                print(f"✗ Error loading {filename}: {str(e)}")
        
        print(f"\n✓ Total documents loaded: {files_loaded}")
        print(f"✓ Total content chunks: {len(documents)}")
        return documents
    
    def create_chunks(self, documents, chunk_size=500, chunk_overlap=50):
        """Split documents into chunks"""
        print(f"\nCreating chunks (size={chunk_size}, overlap={chunk_overlap})...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"✓ Created {len(chunks)} chunks")
        return chunks
    
    def setup_vector_store(self, chunks, config_name="default"):
        """Create embeddings and vector store"""
        print(f"\nSetting up vector store for config: {config_name}...")
        
        if self.embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        persist_path = f"{self.persist_dir}_{config_name}"
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_path
        )
        
        print(f"✓ Vector store created with {len(chunks)} embeddings")
        return self.vectorstore
    
    def setup_qa_chain(self, k=3):
        """Setup QA chain with retrieval"""
        print(f"\nSetting up QA chain (retrieving top-{k} chunks)...")
        
        llm = Ollama(model="mistral", temperature=0.2)
        
        prompt_template = """Answer the question based ONLY on the context provided.
If the answer is not in the context, say "I cannot answer based on the provided documents."

Context: {context}

Question: {question}

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": k}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        print("✓ QA chain ready")
        return self.qa_chain
    
    # ==================== RETRIEVAL METRICS ====================
    
    def calculate_hit_rate(self, retrieved_docs, source_documents) -> float:
        """
        Hit Rate: Whether at least one relevant document was retrieved
        """
        if not source_documents:
            return 1.0  # If no specific source needed, consider it a hit
        
        retrieved_sources = set([doc.metadata.get('source', '') for doc in retrieved_docs])
        expected_sources = set(source_documents)
        
        # Check if any expected source appears in retrieved sources
        hits = any(any(exp in ret for exp in expected_sources) for ret in retrieved_sources)
        return 1.0 if hits else 0.0
    
    def calculate_mrr(self, retrieved_docs, source_documents) -> float:
        """
        Mean Reciprocal Rank: Rank of first relevant document
        """
        if not source_documents:
            return 1.0
        
        for rank, doc in enumerate(retrieved_docs, 1):
            doc_source = doc.metadata.get('source', '')
            if any(exp in doc_source for exp in source_documents):
                return 1.0 / rank
        
        return 0.0
    
    def calculate_precision_at_k(self, retrieved_docs, source_documents, k=3) -> float:
        """
        Precision@K: Fraction of retrieved docs that are relevant
        """
        if not source_documents:
            return 1.0
        
        relevant_count = 0
        for doc in retrieved_docs[:k]:
            doc_source = doc.metadata.get('source', '')
            if any(exp in doc_source for exp in source_documents):
                relevant_count += 1
        
        return relevant_count / min(k, len(retrieved_docs)) if retrieved_docs else 0.0
    
    # ==================== ANSWER QUALITY METRICS ====================
    
    def calculate_answer_relevance(self, answer: str, question: str) -> float:
        """
        Answer Relevance: Semantic similarity between question and answer
        """
        try:
            q_embedding = self.embeddings.embed_query(question)
            a_embedding = self.embeddings.embed_query(answer)
            
            similarity = cosine_similarity([q_embedding], [a_embedding])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def calculate_faithfulness(self, answer: str, retrieved_docs) -> float:
        """
        Faithfulness: How well the answer is grounded in retrieved context
        Simple heuristic: overlap between answer and source docs
        """
        if not retrieved_docs or not answer:
            return 0.0
        
        answer_lower = answer.lower()
        context = " ".join([doc.page_content.lower() for doc in retrieved_docs])
        
        # Count how many answer words appear in context
        answer_words = set(answer_lower.split())
        context_words = set(context.split())
        
        if not answer_words:
            return 0.0
        
        overlap = len(answer_words.intersection(context_words))
        faithfulness = overlap / len(answer_words)
        
        return faithfulness
    
    def calculate_rouge_l(self, generated: str, reference: str) -> float:
        """
        ROUGE-L: Longest common subsequence based metric
        """
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, generated)
        return scores['rougeL'].fmeasure
    
    # ==================== SEMANTIC METRICS ====================
    
    def calculate_cosine_similarity(self, generated: str, reference: str) -> float:
        """
        Cosine Similarity: Semantic similarity between generated and reference
        """
        try:
            gen_embedding = self.embeddings.embed_query(generated)
            ref_embedding = self.embeddings.embed_query(reference)
            
            similarity = cosine_similarity([gen_embedding], [ref_embedding])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def calculate_bleu_score(self, generated: str, reference: str) -> float:
        """
        BLEU Score: N-gram overlap metric
        """
        try:
            reference_tokens = [reference.lower().split()]
            generated_tokens = generated.lower().split()
            
            smoothing = SmoothingFunction().method1
            score = sentence_bleu(reference_tokens, generated_tokens, 
                                smoothing_function=smoothing)
            return score
        except:
            return 0.0
    
    # ==================== EVALUATION PIPELINE ====================
    
    def evaluate_single_question(self, test_case: Dict, question_num: int) -> Dict:
        """Evaluate a single test question"""
        print(f"\n[{question_num}/25] Evaluating: {test_case['question'][:60]}...")
        
        question = test_case['question']
        ground_truth = test_case['ground_truth']
        source_docs = test_case.get('source_documents', [])
        
        try:
            # Get answer from QA chain
            start_time = time.time()
            result = self.qa_chain({"query": question})
            response_time = time.time() - start_time
            
            generated_answer = result['result']
            retrieved_docs = result.get('source_documents', [])
            
            # Calculate all metrics
            metrics = {
                "question_id": test_case['id'],
                "question": question,
                "generated_answer": generated_answer,
                "ground_truth": ground_truth,
                "response_time": round(response_time, 2),
                
                # Retrieval Metrics
                "hit_rate": self.calculate_hit_rate(retrieved_docs, source_docs),
                "mrr": self.calculate_mrr(retrieved_docs, source_docs),
                "precision_at_k": self.calculate_precision_at_k(retrieved_docs, source_docs),
                
                # Answer Quality Metrics
                "answer_relevance": round(self.calculate_answer_relevance(generated_answer, question), 4),
                "faithfulness": round(self.calculate_faithfulness(generated_answer, retrieved_docs), 4),
                "rouge_l": round(self.calculate_rouge_l(generated_answer, ground_truth), 4),
                
                # Semantic Metrics
                "cosine_similarity": round(self.calculate_cosine_similarity(generated_answer, ground_truth), 4),
                "bleu_score": round(self.calculate_bleu_score(generated_answer, ground_truth), 4),
                
                # Metadata
                "num_retrieved_docs": len(retrieved_docs),
                "answerable": test_case.get('answerable', True),
                "question_type": test_case.get('question_type', 'unknown')
            }
            
            print(f"  ✓ ROUGE-L: {metrics['rouge_l']:.3f} | Cosine: {metrics['cosine_similarity']:.3f} | Hit Rate: {metrics['hit_rate']:.1f}")
            
            return metrics
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            return {
                "question_id": test_case['id'],
                "error": str(e),
                "question": question
            }
    
    def evaluate_configuration(self, test_dataset: List[Dict], chunk_size: int, 
                             chunk_overlap: int, config_name: str) -> Dict:
        """Evaluate RAG system with specific chunking configuration"""
        print(f"\n{'='*60}")
        print(f"EVALUATING CONFIGURATION: {config_name}")
        print(f"Chunk Size: {chunk_size} | Overlap: {chunk_overlap}")
        print(f"{'='*60}")
        
        # Load corpus and create chunks
        documents = self.load_corpus()
        chunks = self.create_chunks(documents, chunk_size, chunk_overlap)
        
        # Setup vector store and QA chain
        self.setup_vector_store(chunks, config_name)
        self.setup_qa_chain(k=3)
        
        # Evaluate all questions
        results = []
        for i, test_case in enumerate(test_dataset, 1):
            result = self.evaluate_single_question(test_case, i)
            results.append(result)
        
        # Aggregate metrics
        aggregate = self.aggregate_results(results, config_name)
        
        return {
            "configuration": {
                "name": config_name,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "num_chunks": len(chunks)
            },
            "aggregate_metrics": aggregate,
            "per_question_results": results
        }
    
    def aggregate_results(self, results: List[Dict], config_name: str) -> Dict:
        """Aggregate metrics across all questions"""
        print(f"\n{'='*60}")
        print(f"AGGREGATE RESULTS - {config_name}")
        print(f"{'='*60}")
        
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {"error": "No valid results"}
        
        aggregate = {
            "total_questions": len(results),
            "successful_evaluations": len(valid_results),
            
            # Retrieval Metrics
            "avg_hit_rate": np.mean([r['hit_rate'] for r in valid_results]),
            "avg_mrr": np.mean([r['mrr'] for r in valid_results]),
            "avg_precision_at_k": np.mean([r['precision_at_k'] for r in valid_results]),
            
            # Answer Quality Metrics
            "avg_answer_relevance": np.mean([r['answer_relevance'] for r in valid_results]),
            "avg_faithfulness": np.mean([r['faithfulness'] for r in valid_results]),
            "avg_rouge_l": np.mean([r['rouge_l'] for r in valid_results]),
            
            # Semantic Metrics
            "avg_cosine_similarity": np.mean([r['cosine_similarity'] for r in valid_results]),
            "avg_bleu_score": np.mean([r['bleu_score'] for r in valid_results]),
            
            # Performance
            "avg_response_time": np.mean([r['response_time'] for r in valid_results])
        }
        
        # Print summary
        print(f"\n📊 RETRIEVAL METRICS:")
        print(f"  Hit Rate:       {aggregate['avg_hit_rate']:.3f}")
        print(f"  MRR:            {aggregate['avg_mrr']:.3f}")
        print(f"  Precision@K:    {aggregate['avg_precision_at_k']:.3f}")
        
        print(f"\n📊 ANSWER QUALITY METRICS:")
        print(f"  Answer Relevance: {aggregate['avg_answer_relevance']:.3f}")
        print(f"  Faithfulness:     {aggregate['avg_faithfulness']:.3f}")
        print(f"  ROUGE-L:          {aggregate['avg_rouge_l']:.3f}")
        
        print(f"\n📊 SEMANTIC METRICS:")
        print(f"  Cosine Similarity: {aggregate['avg_cosine_similarity']:.3f}")
        print(f"  BLEU Score:        {aggregate['avg_bleu_score']:.3f}")
        
        print(f"\n⚡ Performance: {aggregate['avg_response_time']:.2f}s avg response time")
        
        return aggregate


def run_comparative_evaluation():
    """Run evaluation across all chunking strategies"""
    print("\n" + "="*60)
    print("RAG EVALUATION FRAMEWORK - COMPARATIVE ANALYSIS")
    print("="*60)
    
    # Load test dataset
    print("\nLoading test dataset...")
    with open('test_dataset.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    test_questions = test_data['test_questions']
    print(f"✓ Loaded {len(test_questions)} test questions")
    
    # Define chunking configurations
    configurations = [
        {"name": "small_chunks", "chunk_size": 250, "chunk_overlap": 25},
        {"name": "medium_chunks", "chunk_size": 550, "chunk_overlap": 55},
        {"name": "large_chunks", "chunk_size": 900, "chunk_overlap": 90}
    ]
    
    # Run evaluation for each configuration
    all_results = []
    
    for config in configurations:
        evaluator = RAGEvaluator()
        result = evaluator.evaluate_configuration(
            test_questions,
            config['chunk_size'],
            config['chunk_overlap'],
            config['name']
        )
        all_results.append(result)
        
        # Small delay between configurations
        time.sleep(2)
    
    # Save results
    output = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_configurations": len(configurations),
        "configurations_tested": all_results
    }
    
    with open('test_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("✓ Evaluation complete! Results saved to test_results.json")
    print(f"{'='*60}")
    
    return output


if __name__ == "__main__":
    run_comparative_evaluation()