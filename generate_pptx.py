"""
NeuroAgent Master's Capstone Presentation Deck Generator (20-Slide Strict Non-Overlapping Edition)
- Light Mode Theme (#F8FAFC background, #0F172A dark slate text, #1D4ED8 / #059669 / #E11D48 highlights).
- Zero generic boilerplate headings ("What is it? / Why we used it?").
- Rigorous mathematical bounding box fitting (add_fitted_picture):
  Every single diagram/image is mathematically clamped inside its bounding box, guaranteeing 100% ZERO OVERLAP with text cards!
- Full coverage of all 5 empirical GPU ablation studies executed live on NVIDIA GeForce RTX 4060 Laptop GPU:
  1. Slide 8: Chunking Strategy & Overlap Ablation (chunk_by_title vs Fixed vs Recursive)
  2. Slide 9: Biomedical vs. General Embedding Models (PubMedBERT vs MiniLM vs BGE on CUDA)
  3. Slide 10: Retrieval Pipeline Ablation (BM25 vs Dense vs Hybrid RRF vs GPU Cross-Encoder)
  4. Slide 18: Complete 6-Panel Real GPU Ablation Dashboard (all 5 studies overview)
  5. Slide 19: Top-K Context Window (K=1, 3, 5, 10) & GPU LLM Temperature (T=0.0..1.0) + Critic Loop Ablation
"""

import os
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "presentation_assets")


def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.50)

    # Light Mode Color Palette
    BG_LIGHT = RGBColor(248, 250, 252)       # #F8FAFC
    CARD_BG = RGBColor(255, 255, 255)        # #FFFFFF
    TEXT_NAVY = RGBColor(15, 23, 42)         # #0F172A
    TEXT_MUTED = RGBColor(51, 65, 85)        # #334155
    BRIGHT_BLUE = RGBColor(29, 78, 216)      # #1D4ED8
    SKY_BLUE = RGBColor(2, 132, 199)         # #0284C7
    EMERALD = RGBColor(5, 150, 105)          # #059669
    ROSE = RGBColor(225, 29, 72)             # #E11D48
    VIOLET = RGBColor(124, 58, 237)          # #7C3AED

    def set_bg(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_LIGHT
        bg.line.fill.background()

    def add_header(slide, slide_num, part_title, heading_text):
        cat_box = slide.shapes.add_textbox(Inches(0.55), Inches(0.20), Inches(12.2), Inches(0.30))
        tf_cat = cat_box.text_frame
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = f"SLIDE {slide_num:02d}  |  {part_title.upper()}"
        p_cat.font.size = Pt(12)
        p_cat.font.bold = True
        p_cat.font.color.rgb = BRIGHT_BLUE

        title_box = slide.shapes.add_textbox(Inches(0.55), Inches(0.46), Inches(12.2), Inches(0.65))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = heading_text
        p_title.font.size = Pt(26)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_NAVY

    def add_fitted_picture(slide, img_name, max_left_in, max_top_in, max_w_in, max_h_in):
        """Mathematically fits and centers image within (max_w, max_h) bounding box, guaranteeing 0 overlap."""
        img_path = os.path.join(ASSETS_DIR, img_name)
        if not os.path.exists(img_path):
            return None
        with Image.open(img_path) as im:
            px_w, px_h = im.size
        aspect = px_w / px_h
        box_aspect = max_w_in / max_h_in

        if box_aspect > aspect:
            fit_h = max_h_in
            fit_w = max_h_in * aspect
            fit_left = max_left_in + (max_w_in - fit_w) / 2.0
            fit_top = max_top_in
        else:
            fit_w = max_w_in
            fit_h = max_w_in / aspect
            fit_left = max_left_in
            fit_top = max_top_in + (max_h_in - fit_h) / 2.0

        return slide.shapes.add_picture(
            img_path,
            Inches(fit_left),
            Inches(fit_top),
            width=Inches(fit_w),
            height=Inches(fit_h)
        )

    def add_bullet_card(slide, left, top, width, height, title, bullets, accent=BRIGHT_BLUE, body_pt=14.5, title_pt=17.0):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = accent
        card.line.width = Pt(1.8)

        tb = slide.shapes.add_textbox(left + Inches(0.20), top + Inches(0.12), width - Inches(0.40), height - Inches(0.24))
        tf = tb.text_frame
        tf.word_wrap = True

        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(title_pt)
        p_t.font.bold = True
        p_t.font.color.rgb = accent
        p_t.space_after = Pt(6)

        for b in bullets:
            p = tf.add_paragraph()
            p.text = f"• {b}"
            p.font.size = Pt(body_pt)
            p.font.color.rgb = TEXT_NAVY
            p.space_after = Pt(4)

    def add_tech_pair_cards(slide, top, left_title, left_bullets, right_title, right_bullets,
                            left_accent=BRIGHT_BLUE, right_accent=EMERALD, height=Inches(2.75), body_pt=14.0, title_pt=16.5):
        add_bullet_card(slide, Inches(0.55), top, Inches(5.95), height, left_title, left_bullets,
                        accent=left_accent, body_pt=body_pt, title_pt=title_pt)
        add_bullet_card(slide, Inches(6.80), top, Inches(5.98), height, right_title, right_bullets,
                        accent=right_accent, body_pt=body_pt, title_pt=title_pt)

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s1)

    banner = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(0.75), Inches(11.9), Inches(6.0))
    banner.fill.solid()
    banner.fill.fore_color.rgb = CARD_BG
    banner.line.color.rgb = BRIGHT_BLUE
    banner.line.width = Pt(3.0)

    tb1 = s1.shapes.add_textbox(Inches(1.15), Inches(1.1), Inches(11.0), Inches(5.3))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "MASTER'S CAPSTONE PROJECT  |  MULTIMODAL MEDICAL AI & AGENTIC RAG"
    p0.font.size = Pt(16)
    p0.font.bold = True
    p0.font.color.rgb = BRIGHT_BLUE
    p0.space_after = Pt(14)

    p1 = tf1.add_paragraph()
    p1.text = "NeuroAgent: 3D Volumetric Perception, Domain-Specific Modular RAG & LangGraph Multi-Agent Architecture for Stroke Neuro-Radiology"
    p1.font.size = Pt(32)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_NAVY
    p1.space_after = Pt(18)

    p2 = tf1.add_paragraph()
    p2.text = (
        "• 3D Perception Layer: ACS-UNet (309,387 Params, 5-Fold Ensemble on ATLAS v3.0 + Healthy Control)\n"
        "• Modular RAG Pipeline: NeuML/pubmedbert-base-embeddings (768-d) + ChromaDB HNSW over 13 Clinical Guidelines\n"
        "• LangGraph Multi-Agent System: StateGraph(AgentState) with Llama-3-8B-Instruct (Ollama) & GPT-4o-mini Critic\n"
        "• Quantitative GPU Ablations: 5 Empirical Studies Executed Live on NVIDIA GeForce RTX 4060 GPU"
    )
    p2.font.size = Pt(18.5)
    p2.font.color.rgb = TEXT_MUTED
    p2.space_after = Pt(22)

    p3 = tf1.add_paragraph()
    p3.text = "Authors: Saumya Prasad & Shambhavi Prasad  |  NMIMS MPSTME (2026)"
    p3.font.size = Pt(19)
    p3.font.bold = True
    p3.font.color.rgb = SKY_BLUE

    # =========================================================================
    # SLIDE 2: CLINICAL & GENERATIVE AI PROBLEM FORMULATION
    # =========================================================================
    s2 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s2)
    add_header(s2, 2, "Part 1: Architecture & Problem Formulation", "Clinical Neuroimaging Bottlenecks & LLM Failure Modes")

    add_bullet_card(s2, Inches(0.55), Inches(1.35), Inches(5.95), Inches(5.65),
                    "1. Clinical & 3D Computer Vision Bottlenecks",
                    [
                        "Manual Planimetry Latency: 3D ABC/2 volumetric delineation requires 15–25 mins per scan during the 4.5h hyperacute window.",
                        "Sub-5 cm³ Lacunar Ambiguity: Chronic small-vessel lacunes mimic benign Virchow-Robin perivascular spaces on T1w MRI.",
                        "Compute & VRAM Ceiling: Standard 3D baselines (nnU-Net: 31.2M params; Swin-UNETR: 62.2M params) require >14 GB VRAM.",
                        "Semantic Gap: Segmentation models output raw binary NIfTI arrays rather than structured ACR clinical reports."
                    ], accent=ROSE, body_pt=16.5, title_pt=19)

    add_bullet_card(s2, Inches(6.80), Inches(1.35), Inches(5.98), Inches(5.65),
                    "2. Generative AI & Clinical Safety Failure Modes",
                    [
                        "Numerical Token Drift: Autoregressive LLMs alter decimal precision (e.g., 80.924 cm³ → 8.09 cm³) under unconstrained decoding.",
                        "Catastrophic Triage Errors: Recommending IV-rtPA when core volume ≥50 cm³ or onset >4.5h risks fatal intracerebral hemorrhage.",
                        "Parametric Guideline Staleness: Pre-trained LLM weights lack updated 2026 AHA/ASA, STRIVE-2, and ESO threshold criteria.",
                        "Zero Provenance: Uncited black-box prose cannot be audited by attending neuroradiologists."
                    ], accent=BRIGHT_BLUE, body_pt=16.5, title_pt=19)

    # =========================================================================
    # SLIDE 3: END-TO-END DECOUPLED SYSTEM ARCHITECTURE
    # =========================================================================
    s3 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s3)
    add_header(s3, 3, "Part 1: Architecture & Problem Formulation", "End-to-End Neuro-Symbolic & Agentic System Architecture")

    add_fitted_picture(s3, "system_pipeline_flow.png", 0.55, 1.20, 12.23, 2.75)
    add_tech_pair_cards(
        s3, Inches(4.25),
        "Stage 1 & 2: 3D Perception & Non-Parametric RAG",
        [
            "ACS-UNet (309k params): Maps 1×192³ NIfTI volume to exact lesion cm³, Δsym, and ASPECTS (0–10).",
            "ClinicalHistoryParser: Extracts symptom onset hours (t_onset), neurological deficits & anticoagulation flags.",
            "ChromaDB HNSW + PubMedBERT (768-d): Retrieves top-3 peer-reviewed guideline passages in <15 ms."
        ],
        "Stage 3 & 4: LangGraph StateGraph & 5-Gate Critic",
        [
            "LangGraph StateGraph(AgentState): Orchestrates Triage, Librarian, Drafter, and Critic nodes.",
            "Dual-LLM Execution: Local Llama-3-8B-Instruct (Ollama) drafts report; GPT-4o-mini + AST audits safety.",
            "Conditional Cyclic Routing: Automatically loops back to Drafter if any numerical or temporal gate fails."
        ],
        left_accent=BRIGHT_BLUE, right_accent=EMERALD, height=Inches(2.80), body_pt=14.0, title_pt=16.5
    )

    # =========================================================================
    # SLIDE 4: ACS-UNET ARCHITECTURE (STRICT 2-SLIDE LIMIT: 1 OF 2 - LEFT/RIGHT)
    # =========================================================================
    s4 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s4)
    add_header(s4, 4, "Part 2: 3D Perception Layer (1/2)", "ACS-UNet: 3D Attention, Contralateral Symmetry & Efficiency")

    add_fitted_picture(s4, "segmentation_comparison.png", 0.55, 1.30, 5.60, 5.65)
    add_bullet_card(s4, Inches(6.45), Inches(1.30), Inches(6.33), Inches(5.65),
                    "Architectural Formulation & Baseline Comparison",
                    [
                        "Depthwise Separable 3D Convolutions: Factorizes 3×3×3 kernels into depthwise + 1×1×1 pointwise ops (309,387 params; 100× smaller than nnU-Net).",
                        "Contralateral Symmetry Adapter (Δsym): Computes |F_enc(X) − Flip_LR(F_enc(X))| across the mid-sagittal plane to isolate subtle <5 cm³ lacunes.",
                        "Why Not nnU-Net (31.2M) or Swin-UNETR (62.2M)?: Require 14.6–18.4 GB VRAM and lack explicit hemispheric symmetry priors.",
                        "Empirical Performance (ATLAS v3.0, n=1,453): 61.91% 5-Fold Mean Dice (+16.3% small-lesion Dice gain) in 146 ms on 4.2 GB VRAM."
                    ], accent=BRIGHT_BLUE, body_pt=16.0, title_pt=19.0)

    # =========================================================================
    # SLIDE 5: ACS-UNET OUTPUT & BIOMARKER PAYLOAD (STRICT 2-SLIDE LIMIT: 2 OF 2)
    # =========================================================================
    s5 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s5)
    add_header(s5, 5, "Part 2: 3D Perception Layer (2/2)", "Volumetric Biomarker Extraction & Temporal History Parsing")

    add_fitted_picture(s5, "mri_4panel_slice.png", 0.55, 1.20, 12.23, 2.75)
    add_tech_pair_cards(
        s5, Inches(4.25),
        "3D Tensor-to-Biomarker Mathematical Extraction",
        [
            "Physical Volume Integration: V_cm3 = Σ I(P_ens(i,j,k) > 0.5) × (Δx·Δy·Δz)/1000 → 80.924 cm³ (80,924 voxels).",
            "Topographic & Symmetry Metrics: Right Hemisphere (MCA) | Δsym = 0.0884 | Estimated ASPECTS = 2 / 10.",
            "Healthy Brain Control Validation: 0.000 cm³ (0 voxels, ASPECTS = 10/10, 100% specificity on normal MRI)."
        ],
        "Neuro-Symbolic Safety Bridge to LangGraph",
        [
            "ClinicalHistoryParser: Regex/NLP parser extracts onset hours (t_onset = 2.0h vs 10.0h vs Wake-Up) & anticoagulant use.",
            "Combined Reperfusion Logic: ivt_eligible = (0 < V_cm3 < 50.0) ∧ (t_onset ≤ 4.5h) ∧ (¬Contraindicated).",
            "Deterministic JSON Payload: Serializes biomarkers + temporal flags into AgentState without LLM math."
        ],
        left_accent=BRIGHT_BLUE, right_accent=EMERALD, height=Inches(2.80), body_pt=14.0, title_pt=16.5
    )

    # =========================================================================
    # SLIDE 6: NON-PARAMETRIC RAG VS. PARAMETRIC FINE-TUNING
    # =========================================================================
    s6 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s6)
    add_header(s6, 6, "Part 3: Domain-Specific Modular RAG", "Non-Parametric Memory (RAG) vs. Parametric Fine-Tuning")

    add_fitted_picture(s6, "rag_vs_finetuning.png", 0.55, 1.20, 12.23, 2.75)
    add_tech_pair_cards(
        s6, Inches(4.25),
        "Non-Parametric Generation Formulation",
        [
            "Marginalized Conditional Probability: P(y | x) ≈ Π_t Σ_{z ∈ Top-K} P_ret(z | x) · P_θ(y_t | x, z, y_<t).",
            "Dynamic Corpus Extensibility: New PDFs (e.g., 2026 AHA/ASA or Radiopaedia ASPECTS) are indexed into ChromaDB in seconds without gradient updates.",
            "Explicit Citation Provenance: Binds every clinical recommendation to exact document metadata [1–6]."
        ],
        "Why Parametric Fine-Tuning (SFT / LoRA) Fails Here",
        [
            "Catastrophic Forgetting & Weight Interference: SFT overwrites general medical syntax while memorizing static cutoffs.",
            "Dosage & Threshold Drift: Parametric weights conflate 0.9 mg/kg rtPA, 0.25 mg/kg Tenecteplase, and 4.5h/24h windows.",
            "Non-Auditable Black Box: Fine-tuned weights cannot output verifiable PDF filenames or page numbers."
        ],
        left_accent=EMERALD, right_accent=ROSE, height=Inches(2.80), body_pt=14.0, title_pt=16.5
    )

    # =========================================================================
    # SLIDE 7: 4-STAGE MODULAR RAG VS. NAIVE LINEAR RAG
    # =========================================================================
    s7 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s7)
    add_header(s7, 7, "Part 3: Domain-Specific Modular RAG", "4-Stage Modular RAG Architecture vs. Naive Linear RAG")

    add_fitted_picture(s7, "modular_vs_naive_rag.png", 0.55, 1.20, 12.23, 2.75)
    add_tech_pair_cards(
        s7, Inches(4.25),
        "NeuroAgent 4-Stage Modular RAG Mechanics",
        [
            "Stage 1 (Biomarker Query Compiler): Translates JSON metrics (80.9 cm³, ASPECTS 2, t=2.0h) into targeted queries.",
            "Stage 2 (Volumetric & Temporal Router): Routes <5 cm³ to STRIVE-2 SVD, >15 cm³ to AHA/ESO LVO, and >4.5h to DAWN/DEFUSE-3.",
            "Stage 3 & 4 (Hybrid Search + Reranker): Combines BM25 + PubMedBERT and reranks Top-15 → Top-3 (Context Precision = 0.91)."
        ],
        "Failure Modes of Naive Single-Stage RAG",
        [
            "Cross-Tier Context Contamination: Flat cosine lookup retrieves lacunar DAPT rules for massive MCA infarcts.",
            "Temporal Window Blindness: Naive embeddings fail to distinguish ≤4.5h IVT windows from 6–24h thrombectomy windows.",
            "Quantitative Impact: Modular routing reduces pre-critic clinical drift from 14.6% (Naive RAG) down to 4.2%."
        ],
        left_accent=BRIGHT_BLUE, right_accent=ROSE, height=Inches(2.80), body_pt=14.0, title_pt=16.5
    )

    # =========================================================================
    # SLIDE 8: ABLATION 1 — CHUNKING STRATEGY & OVERLAP ABLATION
    # =========================================================================
    s8 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s8)
    add_header(s8, 8, "Part 3: Domain-Specific Modular RAG", "Ablation 1: Chunking Strategy & Overlap Ablation (13 PDFs)")

    add_fitted_picture(s8, "chunking_comparison_ablation.png", 0.55, 1.20, 12.23, 2.75)
    add_tech_pair_cards(
        s8, Inches(4.25),
        "Unstructured Title-Aware Chunking (chunk_by_title)",
        [
            "Layout-Aware Parsing (PyMuPDF + Unstructured): Classifies PDF blocks into Title, NarrativeText, ListItem, and Table.",
            "Hyperparameters: max_characters=1200, new_after_n_chars=800, combine_text_under_n_chars=300, overlap=150 chars.",
            "Empirical Result: Optimal 1200c/150-overlap achieves 100.0% Recall@3 while keeping index compact (4,678 chunks vs 14,199 in micro)."
        ],
        "Ablation Analysis: Why Fixed & Zero-Overlap Fail",
        [
            "Syntactic Boundary Severing: Fixed 500c (0.8% integrity) and Recursive 800c (1.1% integrity) slice >98% of clinical clauses mid-sentence.",
            "Overlap Ablation (0 vs. 150 chars): Zero-overlap separates section titles from inclusion/exclusion criteria, causing MRR to collapse.",
            "PubMedBERT 512-Token Alignment: 1,200 chars (~260 words) fully preserves multi-clause criteria without trailing token clipping."
        ],
        left_accent=EMERALD, right_accent=BRIGHT_BLUE, height=Inches(2.80), body_pt=14.0, title_pt=16.5
    )

    # =========================================================================
    # SLIDE 9: ABLATION 2 — BIOMEDICAL VS. GENERAL EMBEDDING MODELS
    # =========================================================================
    s9 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s9)
    add_header(s9, 9, "Part 3: Domain-Specific Modular RAG", "Ablation 2: Biomedical vs. General Embeddings & ChromaDB HNSW")

    add_fitted_picture(s9, "embedding_comparison_hnsw.png", 0.55, 1.20, 12.23, 2.75)
    add_tech_pair_cards(
        s9, Inches(4.25),
        "Real GPU Embedding Benchmark (4,836 Chunks on CUDA)",
        [
            "PubMedBERT (768-d BioMed): Achieves 100.0% Recall@3 and MRR = 0.8333 in 7.73 ms on NVIDIA RTX 4060 GPU.",
            "MiniLM-L6-v2 (384-d General): Achieves 87.5% Recall@3 and MRR = 0.7917 (11.03 ms latency).",
            "BGE-small-en (384-d SOTA): Achieves 87.5% Recall@3 and MRR = 0.7500 (fragments clinical acronyms like ASPECTS)."
        ],
        "ChromaDB HNSW Multi-Layer Graph Indexing",
        [
            "Graph Topology: Configured with M=16 bidirectional links and efConstruction=200 for sub-linear O(log N) traversal.",
            "Cosine Distance Metric: L_cos(u, v) = 1 − (u · v)/(||u||_2 ||v||_2), achieving tight 0.22–0.28 distance on top clinical matches.",
            "HIPAA/DPDP Local Compliance: 100% on-device GPU execution with zero cloud API latency or patient PHI network egress."
        ],
        left_accent=VIOLET, right_accent=BRIGHT_BLUE, height=Inches(2.80), body_pt=14.0, title_pt=16.5
    )

    # =========================================================================
    # SLIDE 10: ABLATION 3 — RETRIEVAL PIPELINE (BM25 VS DENSE VS RERANKER)
    # =========================================================================
    s10 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s10)
    add_header(s10, 10, "Part 3: Domain-Specific Modular RAG", "Ablation 3: Retrieval Pipeline (BM25 vs. Dense vs. Reranker)")

    add_fitted_picture(s10, "hybrid_search_rerank.png", 0.55, 1.20, 12.23, 2.75)
    add_tech_pair_cards(
        s10, Inches(4.25),
        "Stage 1: Sparse BM25 + Dense PubMedBERT RRF",
        [
            "Sparse Lexical (Okapi BM25): 75.0% Rec@1, 87.5% Rec@3, MRR = 0.8125 (matches exact tokens: 0.9 mg/kg, ASPECTS).",
            "Dense Semantic (PubMedBERT): 75.0% Rec@1, 100.0% Rec@3, MRR = 0.8333 (captures clinical concept semantics).",
            "Reciprocal Rank Fusion (k=60): Merges sparse and dense rankings into Top-15 candidate pool (MRR = 0.8542)."
        ],
        "Stage 2: Cross-Encoder Joint Attention Reranking",
        [
            "GPU Cross-Encoder (ms-marco-MiniLM-L-6-v2): Evaluates [CLS] Query [SEP] Chunk [SEP] across full multi-head self-attention.",
            "Empirical Benchmark: Elevates Recall@1 to 87.5%, Recall@3 to 100.0%, and MRR to 0.9167 in 51.69 ms on RTX 4060 GPU.",
            "Concept Coverage: Achieves 82.5% clinical concept recall across all multi-guideline stroke questions."
        ],
        left_accent=SKY_BLUE, right_accent=EMERALD, height=Inches(2.80), body_pt=14.0, title_pt=16.5
    )

    # =========================================================================
    # SLIDE 11: LANGGRAPH CYCLIC STATE MACHINE ARCHITECTURE
    # =========================================================================
    s11 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s11)
    add_header(s11, 11, "Part 4: LangGraph Multi-Agent Architecture", "LangGraph StateGraph(AgentState) & Cyclic Routing")

    add_fitted_picture(s11, "langgraph_state_architecture.png", 0.55, 1.20, 12.23, 2.75)
    add_tech_pair_cards(
        s11, Inches(4.25),
        "LangGraph State Machine Formulation",
        [
            "Typed Shared State (AgentState): Immutable-style TypedDict holding biomarkers, parsed_history, evidence_chunks, draft_report, and error_trace.",
            "Isolated Node Functions: Each agent is a pure state transition f_i: AgentState → ΔAgentState, preventing prompt context pollution.",
            "Conditional Cyclic Edges: graph.add_conditional_edges('critic', route_fn, {'rewrite': 'drafter', 'certify': END}) enables closed-loop self-healing."
        ],
        "Why LangGraph Over Linear Chains or Single Prompts",
        [
            "Failure of Linear DAGs (LangChain SequentialChain): Cannot loop backward when an LLM hallucinates a dosage or violates a time window.",
            "Failure of Monolithic Prompts: Stuffing 3D JSON + 13 PDF rules + ACR formatting into one prompt causes instruction forgetting (28.4% drift).",
            "Bounded Convergence Guarantee: Enforces max_reflections = 2 with deterministic AST fallback for 100% termination."
        ],
        left_accent=BRIGHT_BLUE, right_accent=ROSE, height=Inches(2.80), body_pt=14.0, title_pt=16.5
    )

    # =========================================================================
    # SLIDE 12: THE 4 SPECIALIZED LANGGRAPH AGENT NODES
    # =========================================================================
    s12 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s12)
    add_header(s12, 12, "Part 4: LangGraph Multi-Agent Architecture", "Node-Level Specification of the 4 Specialized NeuroAgents")

    add_fitted_picture(s12, "four_agents_langgraph.png", 0.55, 1.20, 12.23, 2.75)
    add_tech_pair_cards(
        s12, Inches(4.25),
        "Nodes 1 & 2: Perception Triage & Modular Librarian",
        [
            "Node 1 — Triage Agent (SegmentationPerceptionAgent): Executes 5-fold ACS-UNet + ClinicalHistoryParser; computes ivt_eligible & evt_eligible.",
            "Node 2 — Librarian Agent (ModularRetrievalAgent): Compiles tier-specific queries and executes Hybrid RAG over ChromaDB (13 PDFs).",
            "State Mutation: Populates state['biomarkers'] and state['evidence_chunks'] ([1], [2], [3] with PDF & page metadata)."
        ],
        "Nodes 3 & 4: Constrained Drafter & Neuro-Symbolic Critic",
        [
            "Node 3 — Drafter Agent (ReportGeneratorAgent): Runs Llama-3-8B-Instruct (T=0.0) to synthesize ACR sections with concise Impression & [1–6] citations.",
            "Node 4 — Critic Agent (CriticReflectorAgent): Audits draft via GPT-4o-mini + 5 deterministic AST/Regex invariants.",
            "Closed-Loop Feedback: Appends specific gate violations to state['error_trace'] if audit_passed == False."
        ],
        left_accent=BRIGHT_BLUE, right_accent=VIOLET, height=Inches(2.80), body_pt=14.0, title_pt=16.5
    )

    # =========================================================================
    # SLIDE 13: HETEROGENEOUS DUAL-LLM SELECTION STRATEGY
    # =========================================================================
    s13 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s13)
    add_header(s13, 13, "Part 4: LangGraph Multi-Agent Architecture", "Heterogeneous Dual-LLM Routing: Llama-3-8B + GPT-4o-mini")

    add_fitted_picture(s13, "dual_llm_architecture.png", 0.55, 1.20, 12.23, 2.75)
    add_tech_pair_cards(
        s13, Inches(4.25),
        "Why Local Llama-3-8B-Instruct (Ollama) for Drafting",
        [
            "Edge Quantization Efficiency: 4-bit GGUF quantization fits in ~4.7 GB VRAM alongside ACS-UNet (4.2 GB) on a single workstation.",
            "Zero PHI Data Leakage: Raw patient history and neuroimaging metadata are processed locally under HIPAA / DPDP compliance.",
            "Concise Clinical Synthesis: Instructed to report exact voxel metrics in Findings and summarize naturally in Impression."
        ],
        "Why Orthogonal GPT-4o-mini + AST for Critic Auditing",
        [
            "Eliminating Self-Preference Bias: When an LLM audits its own output, it shares the same latent token distribution and overlooks its own errors.",
            "Cross-Family Adversarial Verification: Pairing Meta's Llama-3 with OpenAI's GPT-4o-mini + Python AST guarantees orthogonal auditing.",
            "Latency & Cost Optimization: Lightweight verification payload completes in <450 ms with 0.0% final clinical drift."
        ],
        left_accent=VIOLET, right_accent=EMERALD, height=Inches(2.80), body_pt=14.0, title_pt=16.5
    )

    # =========================================================================
    # SLIDE 14: THE 5-GATE CRITIC / REFLECT VERIFICATION LOOP
    # =========================================================================
    s14 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s14)
    add_header(s14, 14, "Part 4: LangGraph Multi-Agent Architecture", "The 5-Gate Neuro-Symbolic Critic & Reflection Loop")

    add_fitted_picture(s14, "critic_reflection_gates.png", 0.55, 1.20, 12.23, 2.75)
    add_tech_pair_cards(
        s14, Inches(4.25),
        "Formal Specification of the 5 Verification Gates",
        [
            "Gates 1 & 2 (Biomarker Fidelity): Verifies exact float/int presence of infarct_volume_cm3 (80.924), voxels (80,924), laterality, and Δsym (0.0884).",
            "Gate 3 (Dual Volumetric-Temporal Safety Invariant): Blocks IV-rtPA if V_cm3 ≥ 50.0 OR t_onset > 4.5h OR anticoagulant flag is active.",
            "Gates 4 & 5 (BIOSE & Provenance): Enforces negative intracranial checklist (no hemorrhage/shift) and valid [1–6] references."
        ],
        "Iterative Self-Correction & Convergence Dynamics",
        [
            "Error Trace Injection: On failure, Critic constructs a formal violation string (e.g., 'Volume discrepancy: expected 80.924 cm³').",
            "Constrained Rewrite: Drafter receives the violation trace in its system prompt on iteration k+1 and repairs the exact clause.",
            "Empirical Convergence: 94% of reports pass on Iteration 1; 100% converge within ≤2 iterations (0.0% residual hallucination)."
        ],
        left_accent=ROSE, right_accent=EMERALD, height=Inches(2.80), body_pt=14.0, title_pt=16.5
    )

    # =========================================================================
    # SLIDE 15: CERTIFIED CLINICAL REPORT OUTPUT (LIGHT-MODE UI)
    # =========================================================================
    s15 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s15)
    add_header(s15, 15, "Part 5: Clinical Output & Quantitative Evaluation", "End-to-End Output: Certified Clinical Report & Dynamic Triage")

    add_fitted_picture(s15, "clinical_report_mockup.png", 0.55, 1.20, 12.23, 2.75)
    add_tech_pair_cards(
        s15, Inches(4.25),
        "Dynamic Clinical History & Biomarker Sensitivity",
        [
            "Real-Time Temporal Re-Triage: Changing onset from 2.0h → 10.0h → 50.0h dynamically switches IVT/EVT eligibility and RAG queries.",
            "Clinico-Radiological Correlation: Automatically flags ipsilateral vs. contralateral motor/aphasia concordance and drug contraindications."
        ],
        "Academic Numbered Provenance ([1]–[6])",
        [
            "Zero Code Leakage: Cites strictly peer-reviewed Knowledge Base sources ([1] AHA/ASA 2026, [2] ESO 2021, [3] Radiopaedia ASPECTS).",
            "Non-Redundant ACR Prose: Exact voxel counts stated once in Findings and summarized clinically in Impression & Recommendations."
        ],
        left_accent=BRIGHT_BLUE, right_accent=EMERALD, height=Inches(2.80), body_pt=14.0, title_pt=16.5
    )

    # =========================================================================
    # SLIDE 16: EVALUATION DATASETS & BENCHMARK PROTOCOL
    # =========================================================================
    s16 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s16)
    add_header(s16, 16, "Part 5: Clinical Output & Quantitative Evaluation", "Multi-Cohort Evaluation Datasets & Ground-Truth Protocol")

    add_bullet_card(s16, Inches(0.55), Inches(1.35), Inches(3.95), Inches(5.65),
                    "1. ATLAS v3.0 + Healthy Control",
                    [
                        "Cohort Scale: 1,453 multi-center 3D T1w stroke MRIs (192×192×192 isotropic 1 mm³).",
                        "Ground Truth: Manual neuroradiologist 3D lesion masks spanning 12 voxels to >150 cm³.",
                        "Healthy Brain Control: Isotropic normal brain reference scan verifying 0.000 cm³ specificity (ASPECTS 10/10).",
                        "Metrics: 5-Fold Cross-Validated Dice, HD95, Volume R², and Small-Lesion (<5 cm³) F1."
                    ], accent=BRIGHT_BLUE, body_pt=15.5, title_pt=18.5)

    add_bullet_card(s16, Inches(4.68), Inches(1.35), Inches(3.95), Inches(5.65),
                    "2. BIOSE Hospital MRI Corpus",
                    [
                        "Clinical Cohort: 40 clinical brain MRI subjects paired with 34 real hospital radiology reports.",
                        "Structural Calibration: Extracted standard ACR section hierarchy (Technique, Findings, Impression).",
                        "Negative Findings Checklist: Ground-truth template for excluding hemorrhage, midline shift, and hydrocephalus.",
                        "Metrics: Structural Completeness (100%) and Clinical Entity Recall."
                    ], accent=SKY_BLUE, body_pt=15.5, title_pt=18.5)

    add_bullet_card(s16, Inches(8.80), Inches(1.35), Inches(3.95), Inches(5.65),
                    "3. RAGAS & Stroke QA Benchmark",
                    [
                        "Curated Test Suite: 50 complex clinical stroke vignettes paired with expert reference answers.",
                        "Scenario Coverage: Hyperacute ≤4.5h, Extended 6–24h, Wake-Up Stroke, Lacunar SVD, and DOAC Contraindications.",
                        "Retrieval Metrics: Recall@1, Recall@3, Mean Reciprocal Rank (MRR), and Cosine Distance.",
                        "RAGAS Metrics: Faithfulness, Answer Relevance, Context Precision, and Context Recall."
                    ], accent=EMERALD, body_pt=15.5, title_pt=18.5)

    # =========================================================================
    # SLIDE 17: QUANTITATIVE RESULTS (RAGAS & HALLUCINATION ABLATION)
    # =========================================================================
    s17 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s17)
    add_header(s17, 17, "Part 5: Clinical Output & Quantitative Evaluation", "Quantitative Results: RAGAS Metrics & Hallucination Ablation")

    add_fitted_picture(s17, "ragas_and_hallucination_plots.png", 0.55, 1.20, 12.23, 2.75)
    add_tech_pair_cards(
        s17, Inches(4.25),
        "RAGAS Retrieval & Generation Performance (n = 50)",
        [
            "Faithfulness (0.96) & Answer Relevance (0.94): Confirms that generated claims are strictly entailed by retrieved ChromaDB passages.",
            "Context Precision (0.91) & Recall (0.93): Driven by Unstructured chunk_by_title (150-char overlap) + Cross-Encoder reranking."
        ],
        "Architectural Hallucination Ablation Study",
        [
            "Progressive Error Reduction: Zero-Shot LLM (28.4% drift) → Naive Flat RAG (14.6%) → Modular RAG without Critic (4.2%).",
            "Closed-Loop Verification Impact: Adding the 5-Gate LangGraph Critic Loop eliminates the remaining 4.2% tail error to 0.0%."
        ],
        left_accent=BRIGHT_BLUE, right_accent=EMERALD, height=Inches(2.80), body_pt=14.0, title_pt=16.5
    )

    # =========================================================================
    # SLIDE 18: 5 EMPIRICAL GPU ABLATION STUDIES (RTX 4060 CUDA DASHBOARD - LEFT/RIGHT)
    # =========================================================================
    s18 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s18)
    add_header(s18, 18, "Part 5: Clinical Output & Quantitative Evaluation", "5 Empirical GPU Ablation Studies (NVIDIA RTX 4060 CUDA)")

    # Left Column: Fitted 6-Panel Dashboard Plot
    add_fitted_picture(s18, "all_5_ablations_dashboard.png", 0.55, 1.30, 7.20, 5.65)

    # Right Column: Technical Findings Card (Strictly non-overlapping from x=8.00 onwards)
    add_bullet_card(s18, Inches(8.00), Inches(1.30), Inches(4.78), Inches(5.65),
                    "Live RTX 4060 GPU Ablation Findings",
                    [
                        "1. Chunking (13 PDFs): Optimal 1200c/150-overlap achieves 100% Recall@3 (4,678 chunks) vs. 0.8% syntactic boundary integrity in Fixed-500c.",
                        "2. Embeddings (4,836 Chunks): PubMedBERT (768-d) achieves 100% Recall@3 & 0.8333 MRR (7.73 ms) vs. MiniLM (0.7917) & BGE (0.7500).",
                        "3. Retrieval Pipeline: BM25 (0.8125 MRR) → Dense (0.8333) → Hybrid RRF (0.8542) → RRF + GPU Cross-Encoder (0.9167 MRR, 87.5% Rec@1).",
                        "4. Top-K Window: K=3 (2,585 chars) achieves 100% Precision & 88.9% F1 without K=10 token bloat (9,125 chars).",
                        "5. Temperature & Critic Loop: T=0.0 preserves 100% exact volume vs. 40% at T=0.7; 5-Gate Critic reduces violations from 80.0% to 0.0%."
                    ], accent=BRIGHT_BLUE, body_pt=14.0, title_pt=17.5)

    # =========================================================================
    # SLIDE 19: ABLATIONS 4 & 5 — TOP-K CONTEXT & GPU LLM TEMPERATURE
    # =========================================================================
    s19 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s19)
    add_header(s19, 19, "Part 5: Clinical Output & Quantitative Evaluation", "Ablations 4 & 5: Context Window & GPU LLM Temperature")

    add_fitted_picture(s19, "topk_and_temperature_plots.png", 0.55, 1.20, 12.23, 2.75)
    add_tech_pair_cards(
        s19, Inches(4.25),
        "Ablation 4: Top-K Context Window Optimization",
        [
            "K=1 Coverage Deficit: Misses secondary guideline perspectives, dropping Clinical Concept Recall to 65.0% (Context F1 = 78.8%).",
            "K=3 Pareto Sweet Spot: Captures 80.0% concept recall with 100.0% precision (F1 = 88.9%) in a compact 2,585-character prompt.",
            "K=5 & K=10 Token Bloat: Context expands to 9,125 chars (3.5× overhead), diluting precision to 90.0% without clinically actionable gain."
        ],
        "Ablation 5: Real GPU Causal LLM Decoding & Safety",
        [
            "Greedy Decoding (T=0.0): Preserves 100.0% exact volumetric and voxel counts on Qwen2.5-0.5B (RTX 4060 GPU latency = 3,974 ms).",
            "Stochastic Drift (T=0.3 to 1.0): Random sampling causes 20%–60% numerical token mutation, corrupting critical triage volumes.",
            "5-Gate Critic Verification: Eliminates 80.0% Zero-Shot violations and 24.0% RAG sampling errors down to 0.0% certified drift."
        ],
        left_accent=BRIGHT_BLUE, right_accent=EMERALD, height=Inches(2.80), body_pt=14.0, title_pt=16.5
    )

    # =========================================================================
    # SLIDE 20: CONCLUSION & FUTURE RESEARCH HORIZONS
    # =========================================================================
    s20 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s20)
    add_header(s20, 20, "Part 5: Clinical Output & Quantitative Evaluation", "Engineering Contributions & Future Research Horizons")

    add_bullet_card(s20, Inches(0.55), Inches(1.35), Inches(5.95), Inches(5.65),
                    "Core Technical Contributions of NeuroAgent",
                    [
                        "1. Edge-Deployable 3D Perception: ACS-UNet (309k params, 4.2 GB VRAM) delivers 61.91% 5-Fold Dice and 146 ms inference with contralateral Δsym.",
                        "2. Biomedical Modular RAG: PubMedBERT (768-d) + ChromaDB HNSW over 13 stroke guidelines achieves 100% Recall@3 and 0.9167 Hybrid MRR.",
                        "3. LangGraph Cyclic Multi-Agent System: Stateful orchestration of Triage, Librarian, Drafter (Llama-3-8B), and Critic (GPT-4o-mini) nodes.",
                        "4. Neuro-Symbolic Safety Verification: 5-gate volumetric + temporal auditing achieves 0.96 RAGAS Faithfulness and 0.0% clinical drift."
                    ], accent=EMERALD, body_pt=16.0, title_pt=19.5)

    add_bullet_card(s20, Inches(6.80), Inches(1.35), Inches(5.98), Inches(5.65),
                    "Future Architectural & Clinical Extensions",
                    [
                        "1. Multiparametric Acute MRI (DWI / ADC / FLAIR): Quantify diffusion-perfusion mismatch and DWI-FLAIR mismatch for wake-up strokes.",
                        "2. Domain-Adapted QLoRA (Rank r=16): Fine-tune Llama-3-8B on 10,000+ MIMIC-IV / BIOSE neuroradiology reports while retaining RAG provenance.",
                        "3. 3D Counterfactual Diffusion Inpainting: Synthesize pseudo-healthy pre-stroke anatomy via 3D Latent Diffusion to replace flip-based Δsym.",
                        "4. DICOMweb & HL7 FHIR Integration: Package as a containerized PACS microservice for automated emergency stroke code alerts."
                    ], accent=BRIGHT_BLUE, body_pt=16.0, title_pt=19.5)

    target_files = [
        "NeuroAgent_Master_Capstone_Deck.pptx",
    ]
    for target in target_files:
        t_path = os.path.join(BASE_DIR, target)
        try:
            prs.save(t_path)
            print(f"[SAVED] Successfully compiled 20-slide Technical Master Deck: {target}")
        except PermissionError:
            print(f"[LOCKED] Could not overwrite {target} because it is currently open in PowerPoint.")


if __name__ == "__main__":
    build_presentation()
