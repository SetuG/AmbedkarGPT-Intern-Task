"""
Results Analysis Script
Analyzes test_results.json and generates detailed insights
"""

import json
import numpy as np
from collections import defaultdict


def load_results(filename='test_results.json'):
    """Load evaluation results"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_configurations(results):
    """Compare performance across chunking strategies"""
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS: CHUNKING STRATEGIES")
    print("="*80)
    
    configs = results['configurations_tested']
    
    # Create comparison table
    print("\n📊 METRIC COMPARISON TABLE")
    print("-" * 80)
    print(f"{'Metric':<25} {'Small (250)':<15} {'Medium (550)':<15} {'Large (900)':<15}")
    print("-" * 80)
    
    metrics_to_compare = [
        ('avg_hit_rate', 'Hit Rate'),
        ('avg_mrr', 'MRR'),
        ('avg_precision_at_k', 'Precision@K'),
        ('avg_answer_relevance', 'Answer Relevance'),
        ('avg_faithfulness', 'Faithfulness'),
        ('avg_rouge_l', 'ROUGE-L'),
        ('avg_cosine_similarity', 'Cosine Similarity'),
        ('avg_bleu_score', 'BLEU Score'),
        ('avg_response_time', 'Response Time (s)')
    ]
    
    for metric_key, metric_name in metrics_to_compare:
        values = [config['aggregate_metrics'][metric_key] for config in configs]
        print(f"{metric_name:<25} {values[0]:<15.3f} {values[1]:<15.3f} {values[2]:<15.3f}")
    
    print("-" * 80)
    
    # Identify best configuration per metric
    print("\n🏆 BEST CONFIGURATION PER METRIC")
    print("-" * 80)
    
    for metric_key, metric_name in metrics_to_compare:
        values = [config['aggregate_metrics'][metric_key] for config in configs]
        config_names = [config['configuration']['name'] for config in configs]
        
        if metric_key == 'avg_response_time':
            best_idx = np.argmin(values)  # Lower is better for response time
        else:
            best_idx = np.argmax(values)  # Higher is better for quality metrics
        
        print(f"{metric_name:<25} → {config_names[best_idx]:<20} ({values[best_idx]:.3f})")
    
    print("-" * 80)


def analyze_question_types(results):
    """Analyze performance by question type"""
    print("\n" + "="*80)
    print("ANALYSIS BY QUESTION TYPE")
    print("="*80)
    
    for config in results['configurations_tested']:
        config_name = config['configuration']['name']
        print(f"\n📋 Configuration: {config_name}")
        print("-" * 80)
        
        # Group by question type
        type_metrics = defaultdict(list)
        
        for result in config['per_question_results']:
            if 'error' not in result:
                q_type = result.get('question_type', 'unknown')
                type_metrics[q_type].append({
                    'rouge_l': result['rouge_l'],
                    'cosine_similarity': result['cosine_similarity'],
                    'faithfulness': result['faithfulness']
                })
        
        # Calculate averages per type
        for q_type, metrics in type_metrics.items():
            avg_rouge = np.mean([m['rouge_l'] for m in metrics])
            avg_cosine = np.mean([m['cosine_similarity'] for m in metrics])
            avg_faith = np.mean([m['faithfulness'] for m in metrics])
            
            print(f"\n  {q_type.upper()}:")
            print(f"    Questions: {len(metrics)}")
            print(f"    Avg ROUGE-L: {avg_rouge:.3f}")
            print(f"    Avg Cosine:  {avg_cosine:.3f}")
            print(f"    Avg Faithfulness: {avg_faith:.3f}")


def identify_failure_modes(results):
    """Identify common failure patterns"""
    print("\n" + "="*80)
    print("FAILURE MODE ANALYSIS")
    print("="*80)
    
    for config in results['configurations_tested']:
        config_name = config['configuration']['name']
        print(f"\n🔍 Configuration: {config_name}")
        print("-" * 80)
        
        low_performers = []
        
        for result in config['per_question_results']:
            if 'error' not in result:
                # Identify questions with low scores
                if (result['rouge_l'] < 0.2 or 
                    result['cosine_similarity'] < 0.5 or
                    result['faithfulness'] < 0.4):
                    low_performers.append({
                        'id': result['question_id'],
                        'question': result['question'][:60] + "...",
                        'rouge_l': result['rouge_l'],
                        'cosine': result['cosine_similarity'],
                        'faith': result['faithfulness']
                    })
        
        if low_performers:
            print(f"\n⚠️  Found {len(low_performers)} low-performing questions:")
            for item in low_performers[:5]:  # Show top 5
                print(f"\n  Q{item['id']}: {item['question']}")
                print(f"    ROUGE-L: {item['rouge_l']:.3f} | Cosine: {item['cosine']:.3f} | Faithfulness: {item['faith']:.3f}")
        else:
            print("\n✓ No significant failures detected")


def generate_recommendations(results):
    """Generate recommendations based on analysis"""
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    configs = results['configurations_tested']
    
    # Calculate overall scores (weighted average)
    overall_scores = []
    for config in configs:
        metrics = config['aggregate_metrics']
        # Weight: 30% retrieval, 40% quality, 30% semantic
        score = (
            0.30 * (metrics['avg_hit_rate'] + metrics['avg_mrr'] + metrics['avg_precision_at_k']) / 3 +
            0.40 * (metrics['avg_answer_relevance'] + metrics['avg_faithfulness'] + metrics['avg_rouge_l']) / 3 +
            0.30 * (metrics['avg_cosine_similarity'] + metrics['avg_bleu_score']) / 2
        )
        overall_scores.append(score)
    
    best_config_idx = np.argmax(overall_scores)
    best_config = configs[best_config_idx]
    
    print(f"\n🎯 OPTIMAL CONFIGURATION:")
    print(f"   Name: {best_config['configuration']['name']}")
    print(f"   Chunk Size: {best_config['configuration']['chunk_size']}")
    print(f"   Chunk Overlap: {best_config['configuration']['chunk_overlap']}")
    print(f"   Overall Score: {overall_scores[best_config_idx]:.3f}")
    
    print("\n💡 KEY INSIGHTS:")
    
    # Retrieval quality
    best_retrieval = max(configs, key=lambda x: x['aggregate_metrics']['avg_hit_rate'])
    print(f"\n1. Best Retrieval Performance:")
    print(f"   → {best_retrieval['configuration']['name']} (Hit Rate: {best_retrieval['aggregate_metrics']['avg_hit_rate']:.3f})")
    
    # Answer quality
    best_quality = max(configs, key=lambda x: x['aggregate_metrics']['avg_rouge_l'])
    print(f"\n2. Best Answer Quality:")
    print(f"   → {best_quality['configuration']['name']} (ROUGE-L: {best_quality['aggregate_metrics']['avg_rouge_l']:.3f})")
    
    # Speed
    fastest = min(configs, key=lambda x: x['aggregate_metrics']['avg_response_time'])
    print(f"\n3. Fastest Response:")
    print(f"   → {fastest['configuration']['name']} ({fastest['aggregate_metrics']['avg_response_time']:.2f}s)")
    
    print("\n📌 RECOMMENDATIONS:")
    print("\n1. For Production Use:")
    print(f"   Use {best_config['configuration']['name']} configuration")
    print(f"   - Balanced performance across all metrics")
    print(f"   - Chunk size: {best_config['configuration']['chunk_size']}")
    
    print("\n2. For Optimization:")
    if best_config['aggregate_metrics']['avg_faithfulness'] < 0.7:
        print("   - Consider adding more context to prompts")
        print("   - Implement re-ranking of retrieved documents")
    
    if best_config['aggregate_metrics']['avg_hit_rate'] < 0.8:
        print("   - Increase number of retrieved documents (k)")
        print("   - Add hybrid search (keyword + semantic)")
    
    print("\n3. For Specific Use Cases:")
    print(f"   - Speed-critical: Use {fastest['configuration']['name']}")
    print(f"   - Quality-critical: Use {best_quality['configuration']['name']}")
    print(f"   - Retrieval-critical: Use {best_retrieval['configuration']['name']}")


def generate_markdown_report(results):
    """Generate detailed markdown report"""
    print("\n📝 Generating results_analysis.md...")
    
    with open('results_analysis.md', 'w', encoding='utf-8') as f:
        f.write("# RAG Evaluation Results - Detailed Analysis\n\n")
        f.write(f"**Evaluation Date:** {results['evaluation_timestamp']}\n\n")
        f.write("---\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        configs = results['configurations_tested']
        overall_scores = []
        for config in configs:
            metrics = config['aggregate_metrics']
            score = (
                0.30 * (metrics['avg_hit_rate'] + metrics['avg_mrr'] + metrics['avg_precision_at_k']) / 3 +
                0.40 * (metrics['avg_answer_relevance'] + metrics['avg_faithfulness'] + metrics['avg_rouge_l']) / 3 +
                0.30 * (metrics['avg_cosine_similarity'] + metrics['avg_bleu_score']) / 2
            )
            overall_scores.append(score)
        
        best_idx = np.argmax(overall_scores)
        best_config = configs[best_idx]
        
        f.write(f"**Best Configuration:** {best_config['configuration']['name']}\n")
        f.write(f"- Chunk Size: {best_config['configuration']['chunk_size']}\n")
        f.write(f"- Chunk Overlap: {best_config['configuration']['chunk_overlap']}\n")
        f.write(f"- Overall Score: {overall_scores[best_idx]:.3f}\n\n")
        
        # Detailed Metrics
        f.write("## Detailed Metrics Comparison\n\n")
        f.write("| Metric | Small Chunks | Medium Chunks | Large Chunks |\n")
        f.write("|--------|--------------|---------------|---------------|\n")
        
        metrics_list = [
            ('avg_hit_rate', 'Hit Rate'),
            ('avg_mrr', 'MRR'),
            ('avg_precision_at_k', 'Precision@K'),
            ('avg_rouge_l', 'ROUGE-L'),
            ('avg_cosine_similarity', 'Cosine Similarity'),
            ('avg_faithfulness', 'Faithfulness')
        ]
        
        for metric_key, metric_name in metrics_list:
            values = [config['aggregate_metrics'][metric_key] for config in configs]
            f.write(f"| {metric_name} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} |\n")
        
        # Recommendations
        f.write("\n## Recommendations\n\n")
        f.write("### 1. Optimal Configuration\n\n")
        f.write(f"Use **{best_config['configuration']['name']}** for production:\n")
        f.write(f"- Provides best balance of retrieval accuracy and answer quality\n")
        f.write(f"- Chunk size: {best_config['configuration']['chunk_size']} characters\n")
        f.write(f"- Chunk overlap: {best_config['configuration']['chunk_overlap']} characters\n\n")
        
        f.write("### 2. Areas for Improvement\n\n")
        if best_config['aggregate_metrics']['avg_hit_rate'] < 0.8:
            f.write("- **Retrieval**: Consider increasing k or adding hybrid search\n")
        if best_config['aggregate_metrics']['avg_faithfulness'] < 0.7:
            f.write("- **Faithfulness**: Implement answer verification and re-ranking\n")
        if best_config['aggregate_metrics']['avg_rouge_l'] < 0.3:
            f.write("- **Answer Quality**: Fine-tune prompt templates for better alignment\n")
        
        f.write("\n---\n\n")
        f.write("*Report generated automatically by analyze_results.py*\n")
    
    print("✓ Report saved to results_analysis.md")


def main():
    """Main analysis function"""
    print("\n" + "="*80)
    print("RAG EVALUATION - RESULTS ANALYSIS")
    print("="*80)
    
    try:
        results = load_results()
        
        compare_configurations(results)
        analyze_question_types(results)
        identify_failure_modes(results)
        generate_recommendations(results)
        generate_markdown_report(results)
        
        print("\n" + "="*80)
        print("✓ Analysis Complete!")
        print("="*80)
        
    except FileNotFoundError:
        print("\nError: test_results.json not found!")
        print("Please run evaluation.py first to generate results.")
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")


if __name__ == "__main__":
    main()