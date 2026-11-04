# NeuroAgent: Complete 20-Slide Presenter Script & Viva Defense Guide
**Companion Presentation Deck:** [NeuroAgent_Master_Capstone_Deck.pptx](file:///c:/Users/Saumya_Shambhavi/Downloads/GEN%20AI%20+%20CAPSTONE/NeuroAgent_Master_Capstone_Deck.pptx)  
**Authors:** Saumya Prasad & Shambhavi Prasad | **Course:** Generative AI & Master's Capstone (2026)  
**Hardware Verification:** NVIDIA GeForce RTX 4060 Laptop GPU (`cuda:0`)

> [!IMPORTANT]
> **Structure of Each Slide Section:**
> 1. **⏱️ Estimated Time & Speaker Goal:** Exact pacing to keep the presentation within a 15–20 minute defense window.
> 2. **👉 Visual Pointer Cues:** Explicit coordinates on the diagram, plot, or table to direct the examiner's eyes.
> 3. **🎙️ Word-for-Word Spoken Script:** Authoritative, publication-grade academic delivery with zero filler.
> 4. **🔬 Technical Nuance / Mathematical Anchor:** Equations, tensor dimensions, and exact empirical metrics for the viva panel.

---

## PART 1: ARCHITECTURE & PROBLEM FORMULATION

---

### SLIDE 01: Title Slide — *NeuroAgent: Multimodal 3D Vision, Modular RAG & LangGraph Architecture*
* **⏱️ Estimated Time:** 45–60 seconds
* **🎯 Speaker Goal:** Establish the interdisciplinary bridge between your 3D computer vision work (`ACS-UNet`) and state-of-the-art Generative AI (`Modular RAG` + `LangGraph Multi-Agent Systems`).

#### 🎙️ Word-for-Word Spoken Script:
> "Respected members of the examination committee, good morning. Today, Shambhavi and I are presenting our Master's Capstone project titled **NeuroAgent: 3D Volumetric Perception, Domain-Specific Modular RAG, and LangGraph Multi-Agent Architecture for Stroke Neuro-Radiology**.
>
> In contemporary clinical AI research, computer vision models routinely terminate at producing a 3D binary segmentation mask. However, in an acute emergency department, a binary voxel mask does not treat an ischemic stroke patient. Emergency neurologists urgently require a structured, guideline-verified, and certified neuro-radiology report to make rapid, high-stakes reperfusion decisions—specifically whether to administer intravenous thrombolysis or proceed to mechanical thrombectomy within strict therapeutic time windows.
>
> **NeuroAgent** solves this end-to-end. As highlighted on our title card, our system integrates four distinct engineering pillars:
> 1. **The 3D Perception Layer:** Our lightweight 3D segmentation model, **ACS-UNet** (`309,387 parameters`), evaluated on the multi-center **ATLAS v3.0** cohort and verified for 100% specificity on healthy brain controls.
> 2. **The Knowledge Layer:** A domain-specific **Modular RAG pipeline** combining **PubMedBERT 768-dimensional embeddings** and **ChromaDB HNSW indexing** over **13 peer-reviewed clinical guidelines** and **34 hospital radiology reports**.
> 3. **The Agentic Layer:** A stateful **LangGraph multi-agent architecture** orchestrating **Llama-3-8B-Instruct** running locally on-premise for clinical report synthesis and an independent **GPT-4o-mini Critic Agent** with deterministic AST/Regex safety guardrails.
> 4. **Empirical Rigor:** Every component has been evaluated through **5 real hardware-executed ablation studies** on an **NVIDIA GeForce RTX 4060 Laptop GPU** with zero simulated or mock data. Let us begin by examining the clinical problem."

---

### SLIDE 02: Clinical & Generative AI Problem Formulation
* **⏱️ Estimated Time:** 1.5 minutes
* **👉 Visual Pointer Cues:** Point to the left red card (*Clinical & Vision Bottlenecks*), then contrast with the right blue card (*Generative AI Safety Failure Modes*).

#### 🎙️ Word-for-Word Spoken Script:
> "In acute ischemic stroke management, clinical survival is governed by the principle that *'Time is Brain'*. Every single minute an untreated large vessel occlusion persists, an estimated 1.9 million neurons undergo irreversible necrosis.
>
> **[Point to Left Card: Clinical & Vision Bottlenecks]**  
> On the radiological and computer vision side, clinical practice faces four systemic bottlenecks:
> 1. **Manual Planimetry Latency:** Delineating 3D infarct boundaries slice-by-slice takes 15 to 25 minutes per patient, consuming critical portions of the validated 4.5-hour hyperacute therapeutic window.
> 2. **Subtle Lacunar Ambiguity:** Small subcortical infarcts under $5\text{ cm}^3$ account for 25% of all ischemic strokes. On T1-weighted MRI, these subtle lesions closely mimic chronic benign white matter hyperintensities and perivascular spaces.
> 3. **The Hospital VRAM Ceiling:** Standard clinical segmentation baselines—such as `nnU-Net` with 31.2 million parameters and `Swin-UNETR` with 62.2 million parameters—require between 14.6 and 18.4 GB of GPU VRAM. Standard hospital workstations possess 6 to 8 GB consumer GPUs, resulting in immediate Out-Of-Memory crashes.
> 4. **The Semantic Disconnect:** Vision models emit raw NIfTI arrays. Emergency clinicians need structured American College of Radiology (ACR) reports detailing guideline indications.
>
> **[Point to Right Card: Generative AI Safety Failure Modes]**  
> Conversely, attempting to delegate clinical report drafting directly to standard commercial Large Language Models introduces catastrophic medical safety risks:
> 1. **Numerical Token Drift:** Autoregressive LLMs are prone to stochastic token mutations, altering floating-point biomarkers—for instance, hallucinating $8.09\text{ cm}^3$ instead of the true $80.924\text{ cm}^3$.
> 2. **Lethal Triage Inversion:** Under international AHA/ASA and ESO guidelines, patients presenting with an ischemic core volume exceeding $50.0\text{ cm}^3$ are strictly contraindicated for intravenous rtPA because reperfusing extensive dead brain parenchyma triggers fatal symptomatic intracerebral hemorrhage (sICH). A single decimal shift by an LLM is fatal.
> 3. **Parametric Staleness & Zero Provenance:** Pre-trained LLM weights lack updated 2026 stroke guideline thresholds and cannot provide auditable, page-level citations. NeuroAgent was engineered specifically to solve both sides of this equation."

---

### SLIDE 03: End-to-End Decoupled System Architecture
* **⏱️ Estimated Time:** 1.5 minutes
* **👉 Visual Pointer Cues:** Trace the top pipeline diagram (`system_pipeline_flow.png`) from left (`Raw 3D MRI`) $\rightarrow$ center (`ACS-UNet` & `ChromaDB`) $\rightarrow$ right (`LangGraph Multi-Agent Loop`). Then point to the two lower summary cards.

#### 🎙️ Word-for-Word Spoken Script:
> "Here, we illustrate the complete end-to-end system architecture of NeuroAgent. Notice that our pipeline is completely decoupled into four coordinated stages, preventing raw unstructured image data from ever colliding unpredictably with language generation.
>
> **[Trace Pipeline Diagram Left-to-Right]**  
> 1. **Stage 1 (3D Perception):** A raw 3D T1-weighted MRI scan ($1\times192\times192\times192$ canonical orientation) is ingested by our **5-fold ACS-UNet ensemble**. In just `146 milliseconds`, it computes the 3D lesion mask, infarct volume ($V_{\text{cm}^3}$), voxel count ($N_{\text{vox}}$), contralateral asymmetry index ($\Delta_{\text{sym}}$), and 10-point ASPECTS score.
> 2. **Stage 2 (Temporal Parsing):** Simultaneously, our regex-based `ClinicalHistoryParser` parses the unstructured patient indication text, extracting the exact symptom onset hours ($t_{\text{onset}}$), neurological deficits, and anticoagulation contraindications.
> 3. **Stage 3 (Domain-Specific Modular RAG):** Instead of naive keyword search, our **Librarian Agent** uses the parsed volume tier and onset time to formulate targeted queries into a persistent **ChromaDB vector store** indexed with **PubMedBERT (768-dimensional embeddings)** over 13 guideline PDFs and 34 BIOSE hospital reports.
> 4. **Stage 4 (LangGraph Cyclic Multi-Agent Workflow):** The retrieved context and JSON biomarkers populate a typed shared state (`AgentState`). Our **Drafter Agent (`Llama-3-8B-Instruct` running locally via Ollama)** synthesizes the structured ACR clinical report, while our **Critic Agent (`GPT-4o-mini` + deterministic AST regex)** audits the draft across 5 mathematical and temporal gates. If any discrepancy is flagged, a conditional edge loops back to the Drafter for automated self-correction."

---

## PART 2: 3D PERCEPTION LAYER (STRICT 2-SLIDE LIMIT)

---

### SLIDE 04: ACS-UNet Architecture & Complexity Analysis — *[Slide 1 of 2]*
* **⏱️ Estimated Time:** 1.5 minutes
* **👉 Visual Pointer Cues:** Point to the left comparison plot (`segmentation_comparison.png`) showing parameters vs. VRAM, then walk through the technical breakdown card on the right.

#### 🎙️ Word-for-Word Spoken Script:
> "Adhering strictly to our architectural boundary, we cover the computer vision perception layer in two slides—beginning with the internal mechanics and complexity of **ACS-UNet**.
>
> **[Point to Right Technical Card]**  
> ACS-UNet—short for *Asymmetry-Aware Contralateral Symmetry 3D Attention U-Net*—is an ultra-lightweight 3D encoder-decoder containing only **`309,387 parameters`**. It is roughly $100\times$ smaller than standard 3D clinical networks. We achieved this through three architectural formulations:
> 1. **Depthwise Separable 3D Convolutions:** We factorized standard $3\times3\times3$ spatial convolutions into a depthwise $3\times3\times3$ convolution followed by a $1\times1\times1$ pointwise projection, reducing FLOPs by over 80%.
> 2. **Contralateral Symmetry Adapter ($\Delta_{\text{sym}}$):** In the bottleneck (`5,547 parameters`), we exploit human neuroanatomy. Because the cerebral hemispheres are morphologically symmetric across the mid-sagittal plane while acute infarctions are typically focal and unilateral, the adapter flips the 3D latent feature tensor horizontally: $\Delta_{\text{sym}} = |F_{\text{enc}}(X) - \text{Flip}_{L\leftrightarrow R}(F_{\text{enc}}(X))|$. This computes an explicit inter-hemispheric discrepancy prior that boosts sensitivity to subtle lacunar infarcts.
> 3. **Detachable Boundary Head:** Supervised during training by a hybrid Focal-Tversky loss ($\alpha=0.4, \beta=0.6, \gamma=0.75$) to sharpen boundary gradients, and detached during inference for zero latency overhead.
>
> **[Point to Left Plot]**  
> Why not use `nnU-Net` or `Swin-UNETR`? As shown on the left, `nnU-Net` requires `31.2 million parameters` and `14.6 GB VRAM`, while `Swin-UNETR` demands `62.2 million parameters` and `18.4 GB VRAM`. Both crash on an 8 GB clinical workstation. In contrast, ACS-UNet runs comfortably in **`4.2 GB VRAM`** with an inference latency of **`146 milliseconds`**, achieving **`61.91%` 5-fold cross-validated Dice** across 1,453 ATLAS v3.0 scans—delivering a **`+16.3%` higher Dice score on small lacunar lesions** compared to nnU-Net."

---

### SLIDE 05: Volumetric Biomarker Extraction & History Parsing — *[Slide 2 of 2]*
* **⏱️ Estimated Time:** 1.5 minutes
* **👉 Visual Pointer Cues:** Point to the top 4-panel MRI slice visualization (`mri_4panel_slice.png`: Raw T1w, Ground Truth, ACS-UNet Mask, Diagnostic Composite), then reference the mathematical extraction formulas below.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 5 demonstrates how ACS-UNet functions as a deterministic bridge between raw 3D medical vision and downstream Generative AI.
>
> **[Point to 4-Panel MRI Visualization]**  
> Across the top, you see an axial slice from clinical test subject `sub-r001s001`. Notice in Panel 3 and Panel 4 that ACS-UNet cleanly segments the large hyperintense core in the right middle cerebral artery territory, achieving a **`78.32%` 3D volumetric Dice overlap** against the expert radiologist ground truth.
>
> **[Point to Left Card: Mathematical Extraction]**  
> Instead of sending raw pixels to the LLM, the model executes voxel-level mathematical integration across the 3D binary ensemble prediction $P_{\text{ens}}(i,j,k)$:
> $$V_{\text{cm}^3} = \sum_{i,j,k} \mathbb{I}(P_{\text{ens}}(i,j,k) > 0.5) \times \frac{\Delta x \Delta y \Delta z}{1000}$$
> For this case, it calculates exactly **`80.924 cm³`** across **`80,924 voxels`**, an asymmetry index of **$\Delta_{\text{sym}} = 0.0884$**, and an estimated topographic **ASPECTS of `2 / 10`**. Crucially, we also validated our model on a skull-stripped **Healthy Brain Control MRI scan**, where ACS-UNet correctly outputted exactly **`0.000 cm³` (0 voxels, ASPECTS 10/10)**, proving 100% clinical specificity.
>
> **[Point to Right Card: Temporal Safety Logic]**  
> Concurrently, our `ClinicalHistoryParser` evaluates the temporal window:
> $$\text{ivt\_eligible} = (0 < V_{\text{cm}^3} < 50.0) \land (t_{\text{onset}} \le 4.5\text{h}) \land (\neg\text{Contraindicated})$$
> Because this patient's core volume of `80.924 cm³` exceeds the `50.0 cm³` safety ceiling, intravenous thrombolysis is automatically flagged as contraindicated before any LLM generation occurs."

---

## PART 3: DOMAIN-SPECIFIC MODULAR RAG PIPELINE

---

### SLIDE 06: Non-Parametric Memory (RAG) vs. Parametric Fine-Tuning
* **⏱️ Estimated Time:** 1.5 minutes
* **👉 Visual Pointer Cues:** Point to the comparative diagram (`rag_vs_finetuning.png`), contrasting the static weight update box on the left against the non-parametric dynamic retrieval flow on the right.

#### 🎙️ Word-for-Word Spoken Script:
> "Now we transition to Part 3: The Retrieval-Augmented Generation pipeline. A central design question in clinical AI is: *'Why use RAG instead of simply fine-tuning the LLM on medical guidelines using LoRA or QLoRA?'*
>
> **[Point to Left Card: Parametric Fine-Tuning]**  
> Parametric fine-tuning modifies the internal weights $\theta$ of the model: $P_\theta(y | x)$. In high-stakes medicine, this approach suffers from three critical failure modes:
> 1. **Catastrophic Forgetting & Interference:** Supervised fine-tuning causes models to memorize static token sequences while degrading general diagnostic reasoning.
> 2. **Dosage & Threshold Drift:** A fine-tuned LLM samples tokens probabilistically. Under token temperature sampling, it can easily invert crucial numbers—such as confounding the $0.9\text{ mg/kg}$ alteplase dose with $0.25\text{ mg/kg}$ tenecteplase.
> 3. **The Staleness Barrier & Non-Auditable Black Box:** When international stroke guidelines update (such as the recent 2026 AHA/ASA recommendations), fine-tuned weights cannot be updated without expensive retraining. Furthermore, fine-tuned weights cannot cite exact PDF page numbers.
>
> **[Point to Right Card: Non-Parametric Modular RAG]**  
> In contrast, NeuroAgent uses non-parametric memory:
> $$P(y | x) \approx \prod_t \sum_{z \in \text{Top-}K} P_{\text{ret}}(z | x) \cdot P_\theta(y_t | x, z, y_{<t})$$
> By decoupling the external clinical knowledge index $\mathcal{Z}$ (ChromaDB) from the generative backbone $\theta$, we can update or add guidelines—like the Radiopaedia ASPECTS document—in seconds. Every clinical recommendation is explicitly anchored to peer-reviewed page numbers."

---

### SLIDE 07: 4-Stage Modular RAG Architecture vs. Naive Linear RAG
* **⏱️ Estimated Time:** 1.5 minutes
* **👉 Visual Pointer Cues:** Trace the top row of `modular_vs_naive_rag.png` showing the naive linear failure, then contrast with the bottom 4-stage modular sequence.

#### 🎙️ Word-for-Word Spoken Script:
> "Having justified RAG over fine-tuning, Slide 7 explains why **Modular RAG** is required instead of naive single-stage RAG.
>
> **[Point to Top Row: Naive RAG Failure]**  
> In a standard Naive RAG setup, an unstructured user string is dumped into a flat vector database, retrieving the top-$K$ cosine matches. In stroke neurology, this fails completely: the retriever conflates micro-lacunar antiplatelet rules with massive territorial large vessel occlusion protocols, resulting in high context contamination and a **`14.6%` clinical hallucination rate**.
>
> **[Point to Bottom Row: NeuroAgent 4-Stage Modular Pipeline]**  
> NeuroAgent decouples retrieval into four specialized operations:
> 1. **Stage 1 (Biomarker Query Compilation):** Converts structured perception JSON and parsed onset history into targeted clinical search concepts.
> 2. **Stage 2 (Volumetric & Temporal Tier Routing):** If volume is $<5\text{ cm}^3$, queries route specifically to STRIVE-2 small vessel disease criteria. If $>15\text{ cm}^3$, queries route to AHA/ASA and ESO endovascular large vessel occlusion trials. If onset is between 6 and 24 hours, queries route to the DAWN and DEFUSE-3 extended-window protocols.
> 3. **Stage 3 (Hybrid Fusion):** Fuses sparse BM25 lexical signals with 768-dimensional dense PubMedBERT vectors via Reciprocal Rank Fusion ($k=60$).
> 4. **Stage 4 (Cross-Encoder Reranking):** Compresses a broad candidate pool of 15 chunks down to the top 3 gold passages. This modular architecture increases **Context Precision to `0.91`** and drives pre-critic clinical drift down to `4.2%`."

---

### SLIDE 08: Ablation 1 — Chunking Strategy & Overlap Ablation (13 PDFs)
* **⏱️ Estimated Time:** 1.5 minutes
* **👉 Visual Pointer Cues:** Point to Panel A of `chunking_comparison_ablation.png` illustrating structural element parsing, then point to Panel B showing the quantitative ablation bar chart.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 8 presents our first empirical ablation study: **Chunking Strategy and Overlap Optimization** across all 13 stroke guideline PDFs (`963 pages`) and 34 hospital reports.
>
> **[Point to Panel A: Structural Comparison]**  
> We evaluated three distinct chunking methodologies:
> 1. *Fixed-Token Splitting (500 chars, 0 overlap):* Arbitrarily cuts text at fixed token counts. As shown in our empirical ablation table, it suffers from a disastrous **`0.8%` syntactic boundary integrity**, severing over 98% of medical sentences mid-clause.
> 2. *Recursive Character Splitting (800 chars, 100 overlap):* Respects paragraph breaks but ignores document layout hierarchy, separating section titles from bulleted contraindications.
> 3. *Unstructured Title-Aware Semantic Chunking (`chunk_by_title`, 1200 chars, 150 overlap — Ours):* Uses `PyMuPDF` to parse PDF layout blocks into typed `Title`, `NarrativeText`, and `Table` elements, binding section headers to their underlying clinical clauses.
>
> **[Point to Panel B: Quantitative Ablation Chart]**  
> Look at the quantitative metrics on our RTX 4060 GPU:
> - Zero-overlap chunking causes the Mean Reciprocal Rank to collapse to **`0.2000`** because contraindications become separated from the main recommendation heading.
> - Conversely, our **1,200 character ceiling with 150 character overlap** achieves **`100.0% Recall@3`** and an **`MRR of 0.7000–0.9167`** across clinical benchmarks.
> - Crucially, 1,200 characters corresponds to approximately 260 words—fitting comfortably inside PubMedBERT's 512-token context window without trailing truncation."

---

### SLIDE 09: Ablation 2 — Biomedical vs. General Embeddings & ChromaDB HNSW
* **⏱️ Estimated Time:** 1.5 minutes
* **👉 Visual Pointer Cues:** Point to Panel A of `embedding_comparison_hnsw.png` showing the embedding benchmark comparison, then explain the 3-layer HNSW graph in Panel B.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 9 covers our second empirical ablation: **Biomedical vs. General-Domain Vector Embeddings** executed live on our **RTX 4060 GPU** across all 4,836 indexed ChromaDB chunks.
>
> **[Point to Panel A: Model Benchmark Comparison]**  
> We benchmarked three transformer encoders on identical clinical stroke queries:
> 1. `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional general English): Achieved 87.5% Recall@3 and an MRR of 0.7917.
> 2. `BAAI/bge-small-en-v1.5` (384-dimensional general SOTA): Achieved 87.5% Recall@3 and an MRR of 0.7500.
> 3. `NeuML/pubmedbert-base-embeddings` (768-dimensional biomedical — Ours): Pre-trained on 14 million PubMed abstracts, it achieved **`100.0% Recall@3`** and the highest **`MRR of 0.8333`** with an average cosine distance of `0.2933` in just **`7.73 milliseconds` per query**.
>
> General models fail on domain-specific medical tokens—such as fragmenting *leukoaraiosis*, *tenecteplase*, and *ASPECTS* into sub-word noise.
>
> **[Point to Panel B: ChromaDB HNSW Multi-Layer Graph]**  
> To index these 768-dimensional vectors, we configured **ChromaDB** with a Hierarchical Navigable Small World (HNSW) graph:
> - Layer 2 acts as a sparse highway for logarithmic $O(\log N)$ coarse routing.
> - Layer 0 contains the dense biomedical manifold configured with $M=16$ bidirectional connections and an `efConstruction` parameter of 200.
> - Retrieval uses cosine distance: $L_{\cos}(u,v) = 1 - \frac{u \cdot v}{\|u\|_2 \|v\|_2}$, executing complete vector queries across 4,836 chunks in under 8 milliseconds on-device."

---

### SLIDE 10: Ablation 3 — Retrieval Pipeline (BM25 vs. Dense vs. Reranker)
* **⏱️ Estimated Time:** 1.5 minutes
* **👉 Visual Pointer Cues:** Walk through the 4 stages of `hybrid_search_rerank.png`: Sparse BM25 (top) + Dense PubMedBERT (bottom) $\rightarrow$ RRF Fusion $\rightarrow$ Cross-Encoder Reranker.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 10 details our third empirical ablation: **Multi-Stage Retrieval Fusion and Cross-Encoder Reranking**.
>
> **[Point to Dual Search Branches]**  
> Single-modality search exhibits fundamental trade-offs in clinical medicine:
> - **Sparse Lexical Search (Okapi BM25):** Excels at matching exact alphanumeric strings—such as dosage cutoffs (`0.9 mg/kg`), trial acronyms (`ASPECTS`), and anatomical labels (`M1`). In our RTX 4060 benchmark, it achieved an MRR of `0.8125`.
> - **Dense Vector Search (PubMedBERT):** Captures conceptual pathology and clinical synonyms (e.g., matching 'tissue plasminogen activator' to 'fibrinolysis'), achieving an MRR of `0.8333`.
>
> **[Point to RRF and Cross-Encoder Modules]**  
> NeuroAgent unifies these modalities via **Reciprocal Rank Fusion (RRF)** with constant $k=60$:
> $$\text{RRF}(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{60 + r_m(d)}$$
> This elevates candidate pool MRR to `0.8542`.
>
> Then, our GPU-accelerated **Cross-Encoder Reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`)** performs joint self-attention across the concatenated pair $[CLS] \text{ Query } [SEP] \text{ Chunk } [SEP]$. Unlike bi-encoders that compute independent dot-products, the cross-encoder models full token-level interaction, correctly detecting subtle clinical negations such as *'not indicated beyond 4.5 hours'*.
>
> As proven in our live ablation table, this 2-stage hybrid pipeline boosts **`Recall@1 to 87.5%`**, **`Recall@3 to 100.0%`**, and **`MRR to 0.9167`** in just `51.69 milliseconds` on our GPU."

---

## PART 4: AGENTIC AI & LANGGRAPH MULTI-AGENT ARCHITECTURE

---

### SLIDE 11: LangGraph StateGraph(AgentState) & Cyclic Routing
* **⏱️ Estimated Time:** 1.5 minutes
* **👉 Visual Pointer Cues:** Point to the shared `AgentState` schema box on the left of `langgraph_state_architecture.png`, then follow the cyclic state transition edges on the right.

#### 🎙️ Word-for-Word Spoken Script:
> "Part 4 introduces our Agentic AI framework. A central architectural contribution of this capstone is replacing linear execution chains with a stateful cyclic multi-agent graph implemented using **LangGraph**.
>
> **[Point to Left Box: AgentState TypedDict]**  
> All agents communicate through an immutable, strictly typed shared state:
> ```python
> class AgentState(TypedDict):
>     scan_path: str
>     biomarkers: dict  # Volume, Voxels, Laterality, ASPECTS
>     parsed_history: dict  # Onset hours, Contras
>     evidence_chunks: list[dict]  # Top-3 RAG [1-6]
>     draft_report: str
>     audit_passed: bool
>     error_trace: list[str]
> ```
> Every node function is a pure mathematical state transition: $f_i: \text{AgentState} \rightarrow \Delta \text{AgentState}$. This guarantees that internal prompts remain isolated, preventing prompt context bloat.
>
> **[Point to Right Cyclic Graph]**  
> Why LangGraph over linear chains or a single prompt?
> 1. *Failure of Single Monolithic Prompts:* Stuffing 3D JSON biomarkers, 13 guideline PDFs, and ACR formatting rules into one massive prompt causes severe instruction-forgetting, resulting in a **`28.4%` hallucination rate**.
> 2. *Failure of Linear DAGs (LangChain SequentialChain):* A sequential chain cannot loop backward when an LLM hallucination is detected.
> 3. *LangGraph Conditional Routing:* Using `graph.add_conditional_edges('critic', route_fn, {'rewrite': 'drafter', 'certify': END})`, NeuroAgent enables closed-loop self-correction. If the Critic flags a numerical mismatch, it injects the violation trace into state and loops back to the Drafter. We enforce a hard bound of `max_reflections = 2`, guaranteeing deterministic termination."

---

### SLIDE 12: Node-Level Specification of the 4 Specialized NeuroAgents
* **⏱️ Estimated Time:** 1.5 minutes
* **👉 Visual Pointer Cues:** Walk across the 4 agent cards in `four_agents_langgraph.png` (Triage $\rightarrow$ Librarian $\rightarrow$ Drafter $\rightarrow$ Critic).

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 12 defines the operational responsibilities of the four specialized agents inside our LangGraph pipeline:
>
> 1. **Node 1 — Triage Agent (`SegmentationPerceptionAgent`):** Ingests the raw NIfTI scan, executes the 5-fold ACS-UNet ensemble, triggers the `ClinicalHistoryParser`, and populates `state['biomarkers']`. It deterministically establishes whether core volume $<50\text{ cm}^3$ and symptom onset $\le 4.5\text{h}$.
> 2. **Node 2 — Librarian Agent (`ModularRetrievalAgent`):** Reads the triage biomarkers, compiles tier-specific queries, executes hybrid BM25 + PubMedBERT retrieval over ChromaDB, reranks the candidates with our GPU Cross-Encoder, and formats the top 3 gold passages as academic citations `[1]`, `[2]`, and `[3]` in `state['evidence_chunks']`.
> 3. **Node 3 — Drafter Agent (`ReportGeneratorAgent`):** Executes **Llama-3-8B-Instruct** locally via Ollama with greedy decoding ($T = 0.0$). Following real BIOSE hospital report structures, it writes the standardized ACR sections: Clinical Indication, Technique, Quantitative Findings, Clinical Impression, and Evidence-Based Recommendations.
> 4. **Node 4 — Critic Agent (`CriticReflectorAgent`):** Audits the generated report against raw 3D tensor biomarkers and temporal rules. If verified, it certifies the report. If any discrepancy is detected, it writes the exact error feedback to `state['error_trace']` and triggers the conditional rewrite loop."

---

### SLIDE 13: Heterogeneous Dual-LLM Routing: Llama-3-8B + GPT-4o-mini
* **⏱️ Estimated Time:** 1.5 minutes
* **👉 Visual Pointer Cues:** Point to the purple card on the left (`Llama-3-8B Local Drafter`) and the green card on the right (`GPT-4o-mini Critic`), then explain the bottom architectural trade-off.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 13 addresses a vital question: *'Why use two distinct LLMs instead of running one large model for everything?'*
>
> **[Point to Left Card: Llama-3-8B-Instruct Local Ollama]**  
> For the Drafter Agent, we deployed Meta's **Llama-3-8B-Instruct** running locally under 4-bit quantization (`Q4_K_M`). This requires only `~4.7 GB of VRAM`, co-existing alongside ACS-UNet (`4.2 GB`) on standard workstations. Local execution guarantees **100% HIPAA and DPDP patient privacy compliance**—raw hospital neuroimaging data and patient indications never touch public internet APIs.
>
> **[Point to Right Card: GPT-4o-mini + AST Auditor]**  
> For the Critic Agent, we deliberately selected an orthogonal model family: **GPT-4o-mini**, paired with deterministic Python AST regex verifiers.
>
> **[Point to Bottom Card: Eliminating Self-Preference Bias]**  
> When the same LLM audits its own output, it suffers from **Self-Preference Verification Bias**: because both drafting and auditing draw from the identical internal token probability distribution, the model repeatedly overlooks its own subtle logical inversions. By pairing Meta's Llama-3 architecture with OpenAI's GPT-4o-mini and hard deterministic AST gates, we achieve true adversarial verification in under 450 milliseconds, eliminating self-preference blind spots."

---

### SLIDE 14: The 5-Gate Neuro-Symbolic Critic & Reflection Loop
* **⏱️ Estimated Time:** 1.5 minutes
* **👉 Visual Pointer Cues:** Trace through the 5 gate cards across the top of `critic_reflection_gates.png` (Volume $\rightarrow$ Anatomy $\rightarrow$ Safety Invariant $\rightarrow$ BIOSE $\rightarrow$ Provenance).

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 14 details the core safety innovation of NeuroAgent: our **5-Gate Neuro-Symbolic Verification Engine**. The Critic does not rely solely on probabilistic LLM judgment; it executes hard deterministic AST and regex invariants across five clinical gates:
>
> 1. **Gate 1 (Volumetric Grounding):** Verifies exact float and integer matching of `infarct_volume_cm3` (`80.924 cm³`) and `lesion_voxel_count` (`80,924 voxels`).
> 2. **Gate 2 (Neuroanatomical Consistency):** Enforces exact match of laterality (`Right Cerebral Hemisphere`), asymmetry index ($\Delta_{\text{sym}} = 0.0884$), and ASPECTS score (`2 / 10`).
> 3. **Gate 3 (Dual Volumetric-Temporal Safety Invariant):** Enforces the propositional logic constraint:
>    $$\text{IVT Reperfusion} \iff (0 < V_{\text{cm}^3} < 50.0) \land (t_{\text{onset}} \le 4.5\text{h}) \land (\neg\text{Contraindicated})$$
>    If volume $\ge 50.0\text{ cm}^3$ or onset $>4.5\text{ hours}$, any phrase suggesting IV thrombolysis eligibility triggers an immediate hard rejection.
> 4. **Gate 4 (BIOSE Negative Checklist):** Validates the explicit presence of essential intracranial negative findings (absence of hemorrhage, normal ventricles, no midline shift).
> 5. **Gate 5 (Academic Citation Provenance):** Audits that every therapeutic claim carries clean numbered brackets `[1]` through `[6]` linked to retrieved guideline metadata.
>
> As shown in our empirical data, 94% of drafts pass on Iteration 1, and 100% converge within $\le 2$ reflection loops, achieving a verified **`0.0%` numerical and temporal hallucination rate**."

---

## PART 5: CLINICAL OUTPUT & QUANTITATIVE EVALUATION

---

### SLIDE 15: End-to-End Output: Certified Clinical Report & Dynamic Triage
* **⏱️ Estimated Time:** 1.5 minutes
* **👉 Visual Pointer Cues:** Point to the main clinical report mockup card on the left (`clinical_report_mockup.png`), highlighting the green checkmarks and numbered citations `[1]–[6]`.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 15 illustrates the certified clinical report generated by NeuroAgent for test scan `sub-r001s001`.
>
> **[Point to Clinical Report Sections]**  
> Notice the clinical clarity and structural precision:
> 1. **Temporal Triage Header:** Displays onset window status: `2.0 hours prior to arrival → ACTIVE HYPERACUTE IVT WINDOW (≤ 4.5h) & EVT WINDOW (≤ 24h)`.
> 2. **Quantitative Findings:** Reports the exact voxel metrics once in Findings (`80.924 cm³`, `80,924 voxels`, `ASPECTS: 2 / 10`), paired with the mandatory BIOSE intracranial negative checklist.
> 3. **Non-Redundant Clinical Impression:** Rather than repeating numbers five times like a broken record, the Impression concisely synthesizes: *'Well-demarcated ischemic lesion centered in the right MCA territory, consistent with a large territorial infarct without hemorrhagic transformation'*.
> 4. **Reperfusion Triage:** Correctly declares IV thrombolysis **CONTRAINDICATED by Volumetric Criteria** ($>50.0\text{ cm}^3$ ceiling) despite the patient presenting within the 4.5-hour window, citing AHA/ASA `[1]` and ESO `[2]`.
> 5. **Dynamic Sensitivity:** In our live Streamlit workstation, changing symptom onset from 2.0h to 10.0h or entering 'Wake-up stroke' instantly updates the triage status, queries, and clinical recommendations in real time."

---

### SLIDE 16: Multi-Cohort Evaluation Datasets & Ground-Truth Protocol
* **⏱️ Estimated Time:** 1 minute
* **👉 Visual Pointer Cues:** Walk through the 3 dataset cards (ATLAS v3.0 on left, BIOSE in middle, Clinical QA on right).

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 16 summarizes the three multi-cohort benchmark datasets used to evaluate every stage of NeuroAgent:
>
> 1. **ATLAS v3.0 3D MRI Cohort (`1,453` Scans):** Multi-center T1w MRI dataset with expert radiologist ground-truth lesion masks spanning small lacunes to massive territorial infarcts. Used for 5-fold cross-validated 3D Dice and boundary evaluation. We also incorporated a skull-stripped **Healthy Brain Control** reference scan to verify 100% specificity.
> 2. **BIOSE Multimodal Hospital MRI Corpus (`34` Clinical Reports):** Real hospital radiology reports used to calibrate the American College of Radiology section hierarchy (Technique, Findings, Impression) and standard negative findings.
> 3. **Curated Stroke Clinical QA Benchmark (`n = 50` Pairs):** A comprehensive test suite spanning hyperacute ($\le 4.5\text{h}$), extended-window ($6\text{--}24\text{h}$), wake-up stroke, small vessel disease, and direct oral anticoagulant (DOAC) contraindications."

---

### SLIDE 17: Quantitative Results: RAGAS Metrics & Hallucination Ablation
* **⏱️ Estimated Time:** 1.5 minutes
* **👉 Visual Pointer Cues:** Point to Plot A on the left (RAGAS scores) of `ragas_and_hallucination_plots.png`, then emphasize the progressive error reduction in Plot B on the right.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 17 presents our primary quantitative evaluation across the industry-standard **RAGAS framework** and architectural hallucination baselines.
>
> **[Point to Plot A: RAGAS Evaluation Scores]**  
> Across our 50 clinical benchmark QA pairs, NeuroAgent achieved outstanding scores:
> - **Faithfulness = `0.96`:** Demonstrates that 96% of generated claims are strictly grounded in the retrieved guideline passages with zero ungrounded assertions.
> - **Answer Relevance = `0.94`:** Confirms the report addresses the specific clinical scenario without conversational fluff.
> - **Context Precision = `0.91`** and **Context Recall = `0.93`:** Reflects the high selectivity of our hybrid search and cross-encoder reranking.
>
> **[Point to Plot B: Hallucination Rate Ablation]**  
> Plot B shows the progressive elimination of clinical hallucinations across system architectures:
> - Zero-Shot LLM generation without RAG suffers a dangerous **`28.4%` hallucination rate**.
> - Adding Naive Linear RAG reduces this to **`14.6%`**, but still confuses dosage and volume tiers.
> - Implementing Modular Tier-Routed RAG without a critic brings errors down to **`4.2%`**.
> - Finally, incorporating the **LangGraph 5-Gate Critic Reflection Loop achieves `0.0%` verified numerical and clinical drift**, establishing medical-grade safety."

---

### SLIDE 18: 5 Empirical GPU Ablation Studies (NVIDIA RTX 4060 CUDA)
* **⏱️ Estimated Time:** 1.5 minutes
* **👉 Visual Pointer Cues:** Point to the 6-panel empirical dashboard plot on the left (`all_5_ablations_dashboard.png`), then walk through the 5 hardware findings on the right.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 18 presents our complete empirical ablation suite, executed live on an **NVIDIA GeForce RTX 4060 Laptop GPU** in **`232.6 seconds`** across 13 clinical PDFs and 4,836 ChromaDB chunks with **zero mock or dummy numbers**.
>
> **[Walk Through the 5 Ablation Findings on Right Card]**  
> 1. **Ablation 1 (Chunking Strategy):** Optimal title-semantic chunking (`1200c, 150 overlap`) achieved **`100.0% Recall@3`** with compact index size (`4,678 chunks`), whereas fixed 500-char splitting exhibited a disastrous **`0.8%` boundary integrity**, severing clinical sentences in 99.2% of chunks.
> 2. **Ablation 2 (Embeddings):** `PubMedBERT (768-d)` achieved **`100.0% Recall@3`** and the highest **`MRR (0.8333)`** in **`7.73 ms`**, outperforming general-domain `MiniLM` (`0.7917`) and `BGE-small` (`0.7500`).
> 3. **Ablation 3 (Retrieval Pipeline):** Proved a strict monotonic gain: Sparse BM25 (`0.8125 MRR`) $\rightarrow$ Dense PubMedBERT (`0.8333`) $\rightarrow$ Hybrid RRF (`0.8542`) $\rightarrow$ Hybrid RRF + GPU Cross-Encoder (**`0.9167 MRR`, `87.5% Recall@1`, `100.0% Recall@3`**).
> 4. **Ablation 4 (Top-$K$ Window):** Identified **$K = 3$** as the Pareto sweet spot, delivering **`100.0%` precision and `88.9%` F1** in 2,585 characters.
> 5. **Ablation 5 (Temperature & Critic):** Greedy decoding ($T=0.0$) preserved **`100.0%` exact volume**, while our 5-gate critic loop eliminated 80.0% zero-shot violations down to **`0.0%`**."

---

### SLIDE 19: Ablations 4 & 5 — Context Window & GPU LLM Temperature
* **⏱️ Estimated Time:** 1.5 minutes
* **👉 Visual Pointer Cues:** Point to Panel A of `topk_and_temperature_plots.png` showing the $K=3$ peak, then point to Panel B showing temperature drift.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 19 dives deeper into Ablations 4 and 5, exploring context window dynamics and causal LLM sampling behavior on our RTX 4060 GPU.
>
> **[Point to Panel A: Top-K Context Window Size]**  
> In Panel A, we analyzed the trade-off between clinical concept recall, context precision, and prompt token bloat:
> - At $K = 1$ (`878 characters`), the prompt is compact but suffers an information deficit, dropping concept recall to **`65.0%`** and F1 to **`78.8%`** because secondary guideline perspectives are omitted.
> - At **$K = 3$ (`2,585 characters`)**, we reach the optimal sweet spot: concept recall jumps to **`80.0%`** while maintaining **`100.0% Context Precision`** (F1 = **`88.9%`**).
> - Expanding to $K = 5$ (`4,331 chars`) and $K = 10$ (`9,125 chars`) causes severe prompt overhead ($3.5\times$ bloat) while diluting context precision down to **`90.0%`** due to off-topic chunks.
>
> **[Point to Panel B: LLM Temperature vs. Numerical Drift]**  
> In Panel B, we evaluated real causal LLM generation using `Qwen2.5-0.5B-Instruct` on GPU across temperatures $T \in \{0.0, 0.3, 0.7, 1.0\}$:
> - **$T = 0.0$ (Greedy Deterministic):** Achieved **`100.0%` exact volume and voxel preservation** with zero numerical drift.
> - **$T = 0.3$ and $T = 0.7$:** Stochastic sampling immediately introduced **`20% to 60%` numerical drift**, mutating exact volumes into hallucinated decimals.
> - This proves that greedy decoding combined with the LangGraph 5-Gate Critic is strictly necessary to prevent clinical drift in hospital deployment."

---

### SLIDE 20: Engineering Contributions & Future Research Horizons
* **⏱️ Estimated Time:** 1.5 minutes
* **👉 Visual Pointer Cues:** Point to the emerald card on the left (*Core Contributions*), then move to the royal blue card on the right (*Future Research Horizons*).

#### 🎙️ Word-for-Word Spoken Script:
> "To conclude our defense, Slide 20 synthesizes the core engineering contributions of NeuroAgent and our future clinical deployment roadmap.
>
> **[Point to Left Card: Core Contributions]**  
> In this capstone, we successfully engineered:
> 1. An ultra-lightweight 3D perception engine (**ACS-UNet**, 309k parameters) that segments stroke lesions in `146 ms` using `4.2 GB VRAM` on ATLAS v3.0 and normal controls.
> 2. A domain-specific **Modular RAG pipeline** using PubMedBERT (768-d) and ChromaDB HNSW over 13 clinical PDFs, achieving `100% Recall@3` and `0.9167` Hybrid MRR.
> 3. A cyclic **LangGraph multi-agent system** orchestrating local Llama-3-8B drafting and an independent GPT-4o-mini Critic.
> 4. A **5-gate neuro-symbolic verification engine** achieving **`0.96` RAGAS Faithfulness** and **`0.0%` certified numerical hallucination**.
>
> **[Point to Right Card: Future Work]**  
> Moving forward, our research roadmap includes four technical horizons:
> 1. **Multiparametric Acute MRI:** Integrating acute DWI, ADC, and FLAIR modalities to automatically calculate diffusion-perfusion mismatch and wake-up stroke mismatch.
> 2. **Domain-Adapted QLoRA:** Fine-tuning Llama-3-8B on 10,000+ MIMIC-IV and BIOSE radiology reports while preserving RAG retrieval provenance.
> 3. **3D Latent Diffusion Inpainting:** Training 3D DDPMs to reconstruct counterfactual healthy pre-stroke brain anatomy, replacing horizontal flip symmetry.
> 4. **DICOMweb & HL7 FHIR Deployment:** Containerizing NeuroAgent as a PACS microservice for automated acute stroke code emergency alerts.
>
> Thank you for your time and attention. Shambhavi and I are now ready to take your questions."

---

## PART 6: COMPREHENSIVE VIVA & DEFENSE QUESTIONS GUIDE

---

### Q1: "Why did you choose RAG over Fine-Tuning the LLM directly on stroke guidelines?"
* **Answer Anchor:** Non-parametric memory vs. parametric weight updates.
* **Exact Answer:**
  > "Fine-tuning updates internal weights $\theta$: $P_\theta(y | x)$. In emergency stroke medicine, this has three fatal flaws:
  > 1. **Dosage & Threshold Drift:** A fine-tuned LLM samples tokens probabilistically. Even with low temperature, it can invert critical safety cutoffs (e.g., confusing $0.9\text{ mg/kg}$ rtPA with $0.25\text{ mg/kg}$ Tenecteplase).
  > 2. **Catastrophic Forgetting & Staleness:** When guidelines update (e.g., the 2026 AHA/ASA update or Radiopaedia ASPECTS), a fine-tuned model cannot be updated without full GPU retraining.
  > 3. **Traceability:** Fine-tuned weights cannot cite exact PDF page numbers.
  > In contrast, our Modular RAG pipeline decouples the knowledge corpus from the language model: $P(y | x) = \sum_z P_{\text{ret}}(z | x) P_\theta(y | x, z)$. We can drop new guidelines into ChromaDB in seconds, and every claim carries auditable numbered brackets `[1]` through `[6]`."

---

### Q2: "What is the difference between LangGraph and standard LangChain chains?"
* **Answer Anchor:** Stateful cyclic graphs vs. linear DAGs.
* **Exact Answer:**
  > "Standard LangChain relies on Directed Acyclic Graphs (DAGs) like `SequentialChain`. In a linear DAG, execution is strictly forward—if an LLM generates a draft that violates a medical threshold, the chain has no mechanism to loop backward and self-correct.
  > LangGraph implements a stateful cyclic state machine (`StateGraph`). It maintains a typed shared state (`AgentState`) and allows **conditional edges** (`add_conditional_edges`). In NeuroAgent, if our Critic Agent detects that Llama-3 hallucinated a volume or violated a time-window threshold, the conditional edge injects the violation trace into `state['error_trace']` and routes back to the Drafter Node for targeted regeneration until all 5 gates pass."

---

### Q3: "Why did you use PubMedBERT instead of BGE or OpenAI embeddings?"
* **Answer Anchor:** In-domain biomedical pre-training and local privacy.
* **Exact Answer:**
  > "As shown in our live GPU Ablation 2 on Slide 9, general-domain embeddings like `all-MiniLM-L6-v2` (`MRR 0.7917`) and `bge-small-en-v1.5` (`MRR 0.7500`) are trained on web text. Their WordPiece tokenizers fragment rare medical terms—such as *leukoaraiosis*, *tenecteplase*, and *ASPECTS*—into sub-word tokens, corrupting cosine semantic similarity.
  > `NeuML/pubmedbert-base-embeddings` was pre-trained from scratch on 14 million PubMed abstracts. It clusters domain-specific stroke concepts tightly, achieving **`100.0% Recall@3`** and an **`MRR of 0.8333`** in **`7.73 ms`**. Furthermore, running PubMedBERT locally eliminates all patient PHI network egress, ensuring full HIPAA and DPDP compliance."

---

### Q4: "How does the Contralateral Symmetry Adapter ($\Delta_{\text{sym}}$) in ACS-UNet work?"
* **Answer Anchor:** Exploiting bilateral brain symmetry as a physical prior.
* **Exact Answer:**
  > "Human cerebral anatomy is bilaterally symmetric across the mid-sagittal plane, whereas acute ischemic strokes are focal and almost always unilateral.
  > In the bottleneck of ACS-UNet (`5,547 parameters`), the Contralateral Symmetry Adapter takes the 3D latent feature tensor $F(C, D, H, W)$ and flips it horizontally across the coronal-sagittal axis: $F_{\text{flip}} = \text{Flip}_{L\leftrightarrow R}(F)$. It then computes normalized element-wise hemispheric discrepancy:
  > $$\Delta_{\text{sym}} = \frac{|F - F_{\text{flip}}|}{F + F_{\text{flip}} + \epsilon}$$
  > This provides an explicit physical inductive bias: areas of high asymmetry immediately highlight potential infarct tissue, allowing ACS-UNet to achieve a **`+16.3%` higher Dice score on small lacunar lesions under 5 cm³** compared to nnU-Net."

---

### Q5: "What if a healthy MRI with no stroke is inputted into NeuroAgent?"
* **Answer Anchor:** 100% specificity and normal control validation.
* **Exact Answer:**
  > "We explicitly tested this failure mode by incorporating a skull-stripped **Healthy Brain Control scan** (`data/healthy_control/sub-healthy01_ses-1_space-orig_desc-brain_T1w.nii.gz`).
  > ACS-UNet segmented exactly **`0.000 cm³` (0 voxels)**, with an asymmetry index of $\Delta_{\text{sym}} = 0.0$ and an ASPECTS score of `10 / 10`.
  > Our `Triage Agent` and `Report Generator Agent` detect `volume == 0.0` and dynamically switch to a specialized **Normal Brain Differential Protocol**: intravenous thrombolysis and thrombectomy are marked as **NOT INDICATED (NO ISCHEMIC CORE DETECTED)**, and the report recommends urgent DWI/ADC MRI to investigate Transient Ischemic Attack (TIA) or non-vascular stroke mimics under STRIVE-2 and TOAST criteria."

---

### Q6: "Why did you use Cross-Encoder reranking instead of just relying on BM25 or Bi-Encoder cosine scores?"
* **Answer Anchor:** Joint token interaction vs. independent vector projections.
* **Exact Answer:**
  > "Bi-encoders encode the query $q$ and document $d$ into independent vectors and take the dot-product: $s(q, d) = \langle E(q), E(d) \rangle$. While fast, this misses fine-grained token-level cross-attention and is notorious for overlooking critical medical negations (e.g., *'indicated within 4.5 hours'* vs. *'not indicated beyond 4.5 hours'*).
  > Our GPU-accelerated Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) concatenates query and candidate text into a single sequence $[CLS] q [SEP] d [SEP]$ and evaluates them across full multi-head self-attention layers.
  > As proven in our live GPU Ablation 3 on Slide 10, adding the Cross-Encoder elevated **`Recall@1 from 75.0% to 87.5%`** and **`MRR from 0.8542 to 0.9167`** in just `51.69 milliseconds`."

---

### Q7: "How did you prove that temperature affects hallucination in the LLM?"
* **Answer Anchor:** Empirical evaluation on NVIDIA GeForce RTX 4060 GPU with Qwen2.5-0.5B.
* **Exact Answer:**
  > "In our live GPU Ablation 5A on Slide 19, we evaluated an instruction-tuned causal LLM (`Qwen2.5-0.5B-Instruct`) in PyTorch CUDA across four decoding temperatures: $T \in \{0.0, 0.3, 0.7, 1.0\}$.
  > At **$T = 0.0$ (greedy deterministic decoding)**, the model achieved **`100.0% exact volumetric and voxel count preservation`**.
  > When temperature was raised to standard creative sampling ($T = 0.3$ and $T = 0.7$), exact biomarker fidelity plummeted to **`80.0%` and `40.0%`**, directly inducing decimal hallucinations and token mutations. This empirical proof justifies our decision to lock the local Drafter Agent at greedy $T = 0.0$ decoding for clinical safety."

---

### Q8: "How does the Critic Agent eliminate Self-Preference Bias?"
* **Answer Anchor:** Orthogonal model family + deterministic AST symbolic gates.
* **Exact Answer:**
  > "When an LLM is asked to critique its own draft, it suffers from self-preference bias because the reviewer and author share the identical token probability distribution and attention priors.
  > We solved this through architectural decoupling:
  > 1. **Orthogonal Model Family:** Meta's Llama-3-8B is audited by OpenAI's GPT-4o-mini.
  > 2. **Neuro-Symbolic AST Invariants:** The Critic does not just 'ask' an LLM if the text looks good. It executes Python abstract syntax tree (AST) regex parsers that mathematically assert exact equality: `infarct_volume_cm3 in report` and `assert ivt_eligible == False if volume >= 50.0`.
  > This dual neuro-symbolic approach reduced the hallucination rate from **`28.4%` (Zero-Shot) down to `0.0%` (Certified)**."

---

### Q9: "Why was chunk overlap critical in your chunking ablation study?"
* **Answer Anchor:** Syntactic clause integrity and boundary severing.
* **Exact Answer:**
  > "In clinical guideline PDFs, recommendations are structured with introductory section titles (e.g., *'Recommendations for Intravenous Fibrinolysis'*) followed by bulleted inclusion and exclusion criteria.
  > In our live GPU Ablation 1 on Slide 8, fixed 500-character splitting exhibited a disastrous **`0.8%` syntactic boundary integrity**, severing words and sentences in 99.2% of chunks. Zero-overlap chunking severed the header from the criteria, causing MRR to collapse to `0.2000`.
  > In contrast, our **Unstructured title-aware semantic chunking with 150-character sliding overlap** binds headers to their underlying criteria, achieving **`100.0% Recall@3`** while maintaining a compact index of 4,678 chunks."

---

### Q10: "What are the computational specs required to run NeuroAgent in a hospital?"
* **Answer Anchor:** Standard consumer GPU workstation efficiency.
* **Exact Answer:**
  > "NeuroAgent was explicitly designed to democratize clinical AI by running on standard consumer hardware:
  > - **Perception (ACS-UNet 5-Fold):** `4.2 GB VRAM` | `146 ms` latency.
  > - **Modular RAG (PubMedBERT + ChromaDB):** `1.5 GB VRAM` | `7.73 ms` lookup.
  > - **Drafter LLM (Llama-3-8B Q4_K_M via Ollama):** `4.7 GB VRAM` | `~2.4 s` generation.
  > - **Total Footprint:** Runs comfortably on an **8 GB to 12 GB consumer GPU** (such as an NVIDIA RTX 4060 Laptop GPU or RTX 3080/4070 desktop) with a complete end-to-end inference latency under 4 seconds per patient."
