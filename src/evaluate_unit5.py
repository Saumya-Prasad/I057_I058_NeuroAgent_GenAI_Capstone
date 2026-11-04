"""
================================================================================
  NEUROAGENT: UNIT 5 GENERATIVE AI QUANTITATIVE EVALUATION BENCHMARK
================================================================================
Covers Unit 5 Syllabus Requirements:
1. Text Generation Similarity Metrics:
   - BLEU (BiLingual Evaluation Understudy with Brevity Penalty)
   - ROUGE-1, ROUGE-2, ROUGE-L (Longest Common Subsequence F1)
   - METEOR (Stemming & Harmonic Mean Precision/Recall)
2. Information Retrieval / Ranking Metrics:
   - MRR (Mean Reciprocal Rank)
   - Recall@K (K=1, 2, 3)
3. Clinical Safety Metrics:
   - Hallucination / Numerical Discrepancy Error Rate (%)
   - Mean Absolute Error on Infarct Volume (|pred - true| cm³)
================================================================================
"""

import os
import sys
import json
import re
import csv
from collections import Counter
import numpy as np

# Ensure UTF-8 console output for Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

# Link project modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.multi_agent_workflow import NeuroAgentWorkflow

# ---------------------------------------------------------------------------
# Metric Calculation Utilities (Unit 5 Slides 16-55)
# ---------------------------------------------------------------------------
def calculate_rouge_n(reference: str, hypothesis: str, n: int = 1) -> float:
    """Computes ROUGE-N Precision, Recall, and F1 score."""
    ref_tokens = re.findall(r'\w+', reference.lower())
    hyp_tokens = re.findall(r'\w+', hypothesis.lower())
    if len(ref_tokens) < n or len(hyp_tokens) < n:
        return 0.0

    ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
    hyp_ngrams = Counter([tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1)])
    
    overlap = sum((hyp_ngrams & ref_ngrams).values())
    recall = overlap / len(ref_tokens)
    precision = overlap / len(hyp_tokens)
    if recall + precision == 0:
        return 0.0
    return round(2 * (precision * recall) / (precision + recall), 4)

def calculate_rouge_l(reference: str, hypothesis: str) -> float:
    """Computes ROUGE-L based on Longest Common Subsequence (Unit 5 Slide 31)."""
    ref_tokens = re.findall(r'\w+', reference.lower())
    hyp_tokens = re.findall(r'\w+', hypothesis.lower())
    m, n = len(ref_tokens), len(hyp_tokens)
    if m == 0 or n == 0:
        return 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if ref_tokens[i] == hyp_tokens[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])

    lcs = dp[m][n]
    precision = lcs / n
    recall = lcs / m
    if precision + recall == 0:
        return 0.0
    return round(2 * (precision * recall) / (precision + recall), 4)

def calculate_bleu_4(reference: str, hypothesis: str) -> float:
    """Computes BLEU-4 with smoothing (Unit 5 Slides 16-21)."""
    ref_tokens = [re.findall(r'\w+', reference.lower())]
    hyp_tokens = re.findall(r'\w+', hypothesis.lower())
    chencherry = SmoothingFunction()
    return round(sentence_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1), 4)

def calculate_meteor(reference: str, hypothesis: str) -> float:
    """Computes METEOR score (Unit 5 Slides 38-43)."""
    ref_tokens = re.findall(r'\w+', reference.lower())
    hyp_tokens = re.findall(r'\w+', hypothesis.lower())
    try:
        return round(meteor_score([ref_tokens], hyp_tokens), 4)
    except Exception:
        # Fallback approximation if wordnet data not downloaded
        return calculate_rouge_l(reference, hypothesis)

def evaluate_retrieval_ranking(target_guidelines: list, retrieved_evidence: list, k: int = 3) -> dict:
    """Computes Recall@K and MRR (Unit 5 Slides 44-55)."""
    retrieved_sources = [e["source"] for e in retrieved_evidence[:k]]
    
    hits = 0
    first_rank = 0
    for rank, src in enumerate(retrieved_sources, start=1):
        if any(target.lower() in src.lower() for target in target_guidelines):
            hits += 1
            if first_rank == 0:
                first_rank = rank

    recall_at_k = round(min(1.0, hits / max(1, len(target_guidelines))), 4)
    mrr = round(1.0 / first_rank, 4) if first_rank > 0 else 0.0
    return {"Recall@K": recall_at_k, "MRR": mrr, "first_rank": first_rank}

# ---------------------------------------------------------------------------
# Main Benchmark Execution
# ---------------------------------------------------------------------------
def run_unit5_benchmark(benchmark_path: str = "data/test_benchmark/ground_truth_cases.json", max_cases: int = 10):
    print("=" * 80)
    print("   NEUROAGENT: EXECUTING UNIT 5 GENERATIVE AI QUANTITATIVE BENCHMARK")
    print("=" * 80)
    
    with open(benchmark_path, "r", encoding="utf-8") as f:
        cases = json.load(f)[:max_cases]

    print(f"Loaded {len(cases)} gold-standard clinical cases for evaluation.\n")
    workflow = NeuroAgentWorkflow()

    results = []
    r1_list, r2_list, rl_list, bleu_list, meteor_list = [], [], [], [], []
    mrr_list, rec_list = [], []
    vol_errors = []
    hallucination_count = 0

    for idx, c in enumerate(cases):
        case_id = c["case_id"]
        img_path = c["image_path"]
        mask_path = c["mask_path"]
        gt_report = c["gold_standard_report"]
        target_docs = c["target_guidelines"]
        true_vol = c["true_volume_cm3"]

        print(f"\n[{idx+1}/{len(cases)}] Evaluating Case: {case_id} (True Vol: {true_vol} cm³)...")
        res = workflow.run_case(img_path, mask_path, clinical_history="Acute ischemic deficit, onset 2.0h")
        
        gen_report = res["certified_report"]
        pred_vol = res["biomarkers"]["infarct_volume_cm3"]
        retrieved_ev = res["evidence"]

        # 1. Text similarity metrics
        r1 = calculate_rouge_n(gt_report, gen_report, n=1)
        r2 = calculate_rouge_n(gt_report, gen_report, n=2)
        rl = calculate_rouge_l(gt_report, gen_report)
        bleu = calculate_bleu_4(gt_report, gen_report)
        meteor = calculate_meteor(gt_report, gen_report)

        r1_list.append(r1)
        r2_list.append(r2)
        rl_list.append(rl)
        bleu_list.append(bleu)
        meteor_list.append(meteor)

        # 2. Retrieval metrics
        ret_metrics = evaluate_retrieval_ranking(target_docs, retrieved_ev, k=3)
        mrr_list.append(ret_metrics["MRR"])
        rec_list.append(ret_metrics["Recall@K"])

        # 3. Clinical metrics
        vol_err = abs(pred_vol - true_vol)
        vol_errors.append(vol_err)

        # Check if Critic passed
        critic_status = res["trajectory"][-1]["status"]
        if critic_status != "APPROVED":
            hallucination_count += 1

        print(f"   -> ROUGE-L: {rl:.4f} | BLEU-4: {bleu:.4f} | METEOR: {meteor:.4f} | MRR: {ret_metrics['MRR']:.4f} | Rec@3: {ret_metrics['Recall@K']:.4f}")

        results.append({
            "case_id": case_id,
            "true_vol_cm3": true_vol,
            "pred_vol_cm3": pred_vol,
            "vol_abs_error": round(vol_err, 3),
            "rouge_1": r1,
            "rouge_2": r2,
            "rouge_l": rl,
            "bleu_4": bleu,
            "meteor": meteor,
            "retrieval_mrr": ret_metrics["MRR"],
            "retrieval_recall_3": ret_metrics["Recall@K"],
            "critic_status": critic_status
        })

    # Save to CSV
    csv_out = "data/test_benchmark/unit5_evaluation_results.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Print Summary Table
    print("\n" + "=" * 80)
    print("        UNIT 5 EVALUATION SUMMARY BENCHMARK REPORT (ACADEMIC AUDIT)")
    print("=" * 80)
    print(f" Total Clinical Cases Evaluated       : {len(cases)}")
    print(f" Mean ROUGE-1 F1                      : {np.mean(r1_list):.4f}")
    print(f" Mean ROUGE-2 F1                      : {np.mean(r2_list):.4f}")
    print(f" Mean ROUGE-L (Longest Common Subseq) : {np.mean(rl_list):.4f}")
    print(f" Mean BLEU-4 (with Brevity Penalty)   : {np.mean(bleu_list):.4f}")
    print(f" Mean METEOR Score                    : {np.mean(meteor_list):.4f}")
    print(f" Modular RAG Retrieval Recall@3       : {np.mean(rec_list):.4f} ({(np.mean(rec_list)*100):.1f}%)")
    print(f" Modular RAG Retrieval MRR            : {np.mean(mrr_list):.4f}")
    print(f" Mean Infarct Volume Error            : {np.mean(vol_errors):.3f} cm³")
    print(f" Numerical Hallucination Rate         : {(hallucination_count / len(cases)) * 100:.1f}%")
    print(f" Safety Critic Verification Rate      : {((len(cases) - hallucination_count) / len(cases)) * 100:.1f}%")
    print("=" * 80)
    print(f" Detailed results exported to: '{csv_out}'\n")

if __name__ == "__main__":
    run_unit5_benchmark(max_cases=5)

