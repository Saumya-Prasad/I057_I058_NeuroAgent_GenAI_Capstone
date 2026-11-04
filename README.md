# NeuroAgent 🧠
### 3D Volumetric Perception, Domain-Specific Modular RAG & LangGraph Multi-Agent Architecture for Acute Stroke Neuro-Radiology

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2BCUDA-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45.1-FF4B4B.svg)](https://streamlit.io/)
[![Hardware Verified](https://img.shields.io/badge/Hardware-NVIDIA_RTX_4060_GPU-76B900.svg)](https://www.nvidia.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Executive Summary

In acute ischemic stroke triage, survival is governed by the clinical doctrine that **"Time is Brain"**: an untreated large-vessel occlusion destroys **1.9 million neurons every minute**. However, acute clinical care faces two systemic technological bottlenecks:
1. **Vision Models Stop at Binary Masks:** Standard 3D medical vision architectures (`nnU-Net`, `Swin-UNETR`) require prohibitive GPU memory (>14 GB VRAM) and merely output raw binary NIfTI tensors that do not provide actionable treatment plans.
2. **Standard Generative AI Fails in Medicine:** Commercial LLMs suffer from **numerical token drift** (mutating exact lesion volumes like $80.924\text{ cm}^3 \rightarrow 8.09\text{ cm}^3$), **guideline staleness**, and **uncontrolled hallucinations**, risking fatal hemorrhagic transformation if clot-busting medication (IV-rtPA) is recommended outside validated windows.

**NeuroAgent** is an end-to-end, neuro-symbolic clinical system that bridges 3D volumetric image perception and multi-agent Generative AI. It ingests raw 3D patient brain MRIs, extracts deterministic physical biomarkers, performs 4-stage modular retrieval across 13 clinical guidelines, and synthesizes certified American College of Radiology (ACR) neuro-radiology reports via a closed-loop LangGraph state machine.

---

## 🏛️ The Four Architectural Pillars

```
+----------------------------------------------------------------------------------------------------+
|                                    NEUROAGENT ARCHITECTURE                                         |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|   [Patient 3D T1w MRI] --->  ACS-UNet (309k params)  ---> Biomarkers: Vol, Voxels, ASPECTS, Δsym    |
|                                                                    |                               |
|   [Clinical History]   ---> ClinicalHistoryParser    ---> Onset Window (<=4.5h), Contraindications |
|                                                                    |                               |
|                                                                    v                               |
|   [13 Guideline PDFs]  ---> 4-Stage Modular RAG      ---> Top-3 Validated Passages [1]-[6]         |
|                             (BM25 + PubMedBERT + Rerank)           |                               |
|                                                                    v                               |
|   [LangGraph Multi-Agent StateMachine] <===========================> [5-Gate Neuro-Symbolic Critic] |
|   - Triage Agent (Perception & Triage)                               - Volumetric Numerical Match   |
|   - Librarian Agent (ChromaDB HNSW)                                  - Neuroanatomical Laterality   |
|   - Drafter Agent (Llama-3-8B On-Premise)                            - IVT / EVT Safety Invariants  |
|                                                                      - BIOSE Negative Checklist    |
|                                                                      - Academic Bibliography [1-6] |
|                                                                    |                               |
|                                                                    v                               |
|                                    [Certified ACR Neuro-Radiology Report]                          |
+----------------------------------------------------------------------------------------------------+
```

### 1. 3D Volumetric Perception Layer (`ACS-UNet`)
* **Extreme Parameter Efficiency:** Only **309,387 parameters** (over **100× smaller** than `nnU-Net` with 31.2M and `Swin-UNETR` with 62.2M).
* **Hospital Edge-Deployable:** Runs in **4.2 GB VRAM** in **146 ms**, fitting comfortably on 8 GB workstation GPUs.
* **Contralateral Hemispheric Symmetry ($\Delta_{\text{sym}}$):** Computes $|F_{\text{enc}}(X) - \text{Flip}_{LR}(F_{\text{enc}}(X))|$ across the mid-sagittal plane, delivering a **+16.3% Dice improvement on difficult sub-5 cm³ lacunar lesions** on the multi-center **ATLAS v3.0** cohort ($n=1,453$).
* **100% Specificity on Healthy Controls:** Evaluated on normal brains, outputting exactly **0.000 cm³** with zero false-positive hallucinations.

### 2. Domain-Specific Modular RAG (`ChromaDB` + `PubMedBERT`)
* **Title-Aware Semantic Chunking:** Layout-aware parsing via `PyMuPDF` and `Unstructured` (1200c max, 150 overlap) achieves **100.0% Recall@3** across 4,678 chunks, eliminating boundary severing found in fixed-character splitters (0.8% integrity).
* **Biomedical Embeddings on CUDA:** `NeuML/pubmedbert-base-embeddings` (768-d) achieves **0.8333 MRR** in **7.73 ms**, outperforming general-domain `MiniLM` (0.7917) and `BGE-small` (0.7500).
* **Two-Stage Hybrid Search & Reranking:** Sparse BM25 (exact tokens like `0.9 mg/kg`) + Dense PubMedBERT merged via Reciprocal Rank Fusion ($k=60$) and reranked by `cross-encoder/ms-marco-MiniLM-L-6-v2` (**0.9167 MRR, 87.5% Recall@1, 100.0% Recall@3** in 51.69 ms).

### 3. Stateful LangGraph Multi-Agent Architecture
* **Typed `AgentState`:** Orchestrates 4 specialized agents: **Triage Agent**, **Librarian Agent**, **Drafter Agent**, and **Critic Agent**.
* **Heterogeneous Dual-LLM Routing:**
  * **Drafter Agent:** Local, on-premise **Llama-3-8B-Instruct** (Ollama 4-bit GGUF, 4.7 GB VRAM). **Zero Protected Health Information (PHI) egress**, ensuring HIPAA / DPDP regulatory compliance.
  * **Critic Agent:** Independent **GPT-4o-mini + Python AST**, eliminating self-preference bias.
* **5-Gate Reflection Loop:** Audits numerical volume fidelity, laterality, clinical reperfusion safety invariants, BIOSE intracranial negative checklist, and academic citations.
* **Guaranteed Convergence:** **94% pass on Iteration 1**, **100% converge in $\le 2$ iterations** under a hard bounded termination condition ($\text{max\_reflections} \le 2$).

### 4. 100% Real Empirical GPU Ablation Suite
All 5 ablations executed on an **NVIDIA GeForce RTX 4060 Laptop GPU** (`cuda:0`, 8 GB VRAM) across 13 clinical guidelines and 4,836 chunks:
1. **Chunking Strategy Ablation:** Fixed vs. Recursive vs. Title-Aware Chunking (13 PDFs).
2. **Embedding Architecture Ablation:** PubMedBERT vs. MiniLM vs. BGE-small on CUDA.
3. **Retrieval Pipeline Ablation:** Sparse BM25 vs. Dense vs. Hybrid RRF vs. GPU Cross-Encoder.
4. **Top-K Context Window Ablation:** $K \in \{1, 3, 5, 10\}$ Pareto optimization ($K=3$: 100% precision, 88.9% F1).
5. **LLM Temperature & Safety Ablation:** Greedy $T=0.0$ (100% exact numerical preservation) vs. $T=0.7$ (60% drift); Critic reduces safety violations from **80.0% to 0.0%**.

---

## 📊 Quantitative Evaluation Benchmark

### 1. RAGAS Retrieval & Clinical Generation Metrics ($n = 50$)
| Evaluation Metric | Score | Clinical Meaning |
| :--- | :---: | :--- |
| **Faithfulness** | **0.96** | 96% of clinical claims are mathematically entailed by retrieved guideline passages. |
| **Answer Relevance** | **0.94** | Generates direct, actionable triage recommendations without conversational filler. |
| **Context Precision** | **0.91** | Validated clinical passages are ranked at the top of candidate pools. |
| **Context Recall** | **0.93** | Successfully retrieves 93% of required clinical criteria (dosing, contraindications). |

### 2. Progressive Hallucination & Safety Violation Reduction
| Configuration | Clinical Drift / Error Rate |
| :--- | :---: |
| **Baseline Zero-Shot LLM** | `28.4%` |
| **+ Naive Linear RAG** | `14.6%` |
| **+ 4-Stage Modular RAG** | `4.2%` |
| **+ LangGraph 5-Gate Closed-Loop Critic (NeuroAgent)** | **`0.0%`** |

---

## 📁 Repository Structure

```
I057_I058_NeuroAgent_GenAI_Capstone/
├── a04b06_5_folds_best_pth/           # Trained ACS-UNet 5-Fold Model Checkpoints (6 MB total)
│   ├── a04b06_fold_1_best.pth
│   └── ...
├── data/
│   ├── biose_reports/                 # Hospital Brain MRI Clinical Diagnosis Reports
│   ├── healthy_control/               # Sample 3D T1w Scans for Testing (0.0 cm³ Controls)
│   └── test_benchmark/                # Ground Truth Benchmark & Real GPU Ablation JSON
│       ├── all_5_ablations_real_gpu.json
│       ├── ground_truth_cases.json
│       └── unit5_evaluation_results.csv
├── guildines/                         # 13 Peer-Reviewed Clinical Stroke Guidelines (PDFs)
│   ├── Prabhakaran et al. (AHA/ASA 2026 Guideline)
│   ├── Berge et al. (ESO 2021/2024 Thrombolysis)
│   ├── STRIVE-2 (Small Vessel Disease Neuroimaging)
│   ├── Radiopaedia ASPECTS Reference Article
│   └── DAWN & DEFUSE-3 NEJM Thrombectomy Trials
├── src/                               # Core Source Code & GenAI RAG Pipeline
│   ├── multi_agent_workflow.py        # LangGraph Multi-Agent System & 5-Gate Critic
│   ├── perception.py                  # 3D ACS-UNet Perception Engine & Biomarkers
│   ├── ingest_and_chunk_unstructured.py # Title-Aware Semantic Ingestion
│   ├── rag_indexer.py                 # Sliding-Window Chunker & Vector Store Builder
│   ├── run_all_5_ablations_gpu.py     # 100% Real Empirical 5-Ablation GPU Runner
│   ├── evaluate_unit5.py              # BLEU-4, ROUGE-L, METEOR, MRR Benchmark
│   └── ablation_chunking_study.py     # Chunking Ablation Study Script
├── app.py                             # Streamlit Interactive Web Application
├── NeuroAgent_Capstone_Final_Report.md # Full Capstone Final Technical Report
├── requirements.txt                   # Version-Pinned Python Dependencies
├── LICENSE                            # MIT License
└── README.md
```

---

## 🚀 Quickstart & Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Saumya-Prasad/I057_I058_NeuroAgent_GenAI_Capstone.git
cd I057_I058_NeuroAgent_GenAI_Capstone
```

### 2. Create and Activate Virtual Environment
```bash
conda create -n neuroagent python=3.11 -y
conda activate neuroagent
pip install -r requirements.txt
```

### 3. Build the Vector Knowledge Base
To ingest the 13 clinical PDFs and build the ChromaDB vector database locally:
```bash
python src/ingest_and_chunk_unstructured.py
```

### 4. Launch the Streamlit Interactive Copilot
```bash
streamlit run app.py
```
* **Tab 1:** Run end-to-end multi-agent clinical cases with real-time 3D slice visualization, dynamic history parsing, and ACR report generation.
* **Tab 2:** Inspect the real empirical 5-ablation GPU benchmark dashboard and interactive metric tables.

### 5. Reproduce the Real 5-Ablation GPU Suite
To re-run all 5 empirical experiments on your local CUDA GPU:
```bash
python src/run_all_5_ablations_gpu.py
```

---

## 👥 Authors & Academic Capstone

* **Saumya Prasad** ([@Saumya-Prasad](https://github.com/Saumya-Prasad))
* **Shambhavi Prasad** ([@Shambhavi-2112](https://github.com/Shambhavi-2112))
* **Institution:** SVKM's NMIMS MPSTME, Mumbai
* **Course:** Generative AI & Master's Capstone (2026)

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

