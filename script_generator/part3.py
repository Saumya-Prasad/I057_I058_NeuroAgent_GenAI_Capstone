# script_generator/part3.py

def get_part3_text():
    return """
---

## PART 3: DOMAIN-SPECIFIC MODULAR RAG & RETRIEVAL ABLATIONS (SLIDES 11 – 20)

---

### SLIDE 11: Non-Parametric Memory (RAG) vs. Fine-Tuning
* **Slide Title:** *Non-Parametric Memory (RAG) vs. Fine-Tuning*
* **Slide Theme:** Conceptual comparison between static model weights and external vector knowledge retrieval.
* **⏱️ Estimated Time:** 45 seconds
* **👉 Visual Pointer Cues:** Point to the left branch showing the closed-box Fine-Tuned Model (static parametric weights) vs. the right branch showing Retrieval-Augmented Generation (dynamic, external knowledge store).

#### 🎙️ Word-for-Word Spoken Script:
> "Now, let us transition to Part 3: our **Domain-Specific Modular RAG Knowledge Architecture**.
>
> In the Generative AI community, there is a constant debate: when building a specialized domain system, should you fine-tune the model's internal weights via supervised instruction tuning (SFT) or LoRA, or should you deploy Retrieval-Augmented Generation (RAG)?
>
> On Slide 11, we illustrate this fundamental architectural divide. In acute medical healthcare—and especially in stroke neurology—we assert that **fine-tuning static weights is an unacceptable clinical liability**, and that non-parametric external memory through RAG is mathematically and clinically superior. Let us examine the exact theory why."

---

### SLIDE 12: Why RAG Beats Parametric Fine-Tuning
* **Slide Title:** *Why RAG Beats Parametric Fine-Tuning*
* **Cards Displayed:** Left Card (*Non-Parametric Generation [RAG]*) vs. Right Card (*Why SFT / LoRA Fails Here*)
* **⏱️ Estimated Time:** 90 seconds
* **👉 Visual Pointer Cues:** Point to the mathematical marginalization formula on the left card ($P(y|x) \approx \prod \dots$). Then highlight the three failure modes on the right card: Catastrophic Forgetting, Dosage Drift, and Lack of Auditability.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 12 provides the formal mathematical and clinical justification for why RAG fundamentally outperforms parametric fine-tuning:
>
> **[Point to Left Card: Non-Parametric Generation]**  
> Formally, RAG models the probability of generating clinical token sequence $y$ given patient input $x$ by marginalizing over a set of top-$K$ retrieved evidence passages $z$:
> $$P(y|x) \approx \prod_{t=1}^T \sum_{z \in \text{Top-K}} P_{\text{ret}}(z|x) \cdot P_\theta(y_t \mid x, z, y_{<t})$$
> This formulation delivers three decisive clinical advantages:
> 1. **Zero-Latency Extensibility:** When the American Heart Association publishes an updated guideline (such as the 2026 Prabhakaran guideline) or Radiopaedia publishes an ASPECTS update, we index the new PDF into ChromaDB in seconds without training a single neural weight or spending thousands of dollars on GPU clusters.
> 2. **Explicit Lineage and Provenance:** Because $z$ is explicitly retrieved from disk, every medical statement in the report can be tied directly to a specific PDF document and page number.
>
> **[Point to Right Card: Why SFT / LoRA Fails Here]**  
> Conversely, fine-tuning introduces three catastrophic clinical risks:
> 1. **Catastrophic Forgetting:** Supervised fine-tuning overwrites the general medical reasoning and linguistic syntax learned during pre-training.
> 2. **Dosage and Threshold Drift:** Parametric weights store knowledge as diffuse floating-point numbers across billions of connections. When fine-tuned on multiple papers, the model conflates different drug regimens—such as confusing the 0.9 mg/kg dose of Alteplase with the 0.25 mg/kg dose of Tenecteplase, or blending the 4.5-hour IVT window with the 24-hour thrombectomy window.
> 3. **Non-Auditability:** A fine-tuned neural network cannot tell a radiologist which page of which guideline it read to make its recommendation, making clinical certification impossible."

#### 🔬 Plain-English Concept Explanations:
* **Biological / Medical Aspect (Why Clinical Guidelines Keep Changing):**
  * Stroke neurology moves fast. In 2015, clinical trials proved that mechanical thrombectomy worked up to 6 hours. In 2018, the DAWN and DEFUSE-3 trials proved it could work up to 24 hours in selected patients. In 2021, the ESO updated guidelines for wake-up strokes. In 2026, the AHA/ASA published new recommendations on Tenecteplase dosing.
  * If a doctor relied on an AI model trained three years ago, the AI would be dangerously out of date! RAG allows the hospital to simply drop the latest PDF guideline into a folder, and the AI immediately adopts the new medical standard.
* **Technical / Computational Aspect (Open-Book Exam vs. Cramming from Memory):**
  * **Fine-Tuning (Parametric Memory):** Like forcing a medical student to memorize an entire medical library. They will remember general themes, but when stressed, they might mix up a decimal point or confuse two drug dosages.
  * **RAG (Non-Parametric Memory):** Like giving that same student an open-book exam with the exact pages of the official guidelines placed right in front of them. The student doesn't have to guess or memorize; they look at the book, read the exact sentence, and cite the page number.

---

### SLIDE 13: 4-Stage Modular RAG vs. Naive Linear RAG
* **Slide Title:** *4-Stage Modular RAG vs. Naive Linear RAG*
* **Slide Theme:** Visual contrast between naive single-lookup RAG and NeuroAgent's 4-stage modular pipeline.
* **⏱️ Estimated Time:** 45 seconds
* **👉 Visual Pointer Cues:** Trace the top linear flow (Naive RAG: User Query $\rightarrow$ Vector Search $\rightarrow$ LLM) and contrast it with the bottom multi-stage pipeline (Compiler $\rightarrow$ Router $\rightarrow$ Hybrid RRF $\rightarrow$ Cross-Encoder).

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 13 highlights our architectural divergence from standard RAG.
>
> Most commercial applications implement **Naive Linear RAG**: a user types a query, a vector database performs a single cosine-similarity lookup, and the top-5 text chunks are dumped into an LLM context window.
>
> In acute medicine, Naive RAG fails catastrophically due to context contamination and semantic confusion. To solve this, we architected **NeuroAgent 4-Stage Modular RAG**:
> 1. A **Clinical Query Compiler** that translates raw numbers into clinical search concepts;
> 2. A **Dynamic Tier Router** that partitions guideline search space;
> 3. A **Hybrid BM25 and PubMedBERT Retrieval Engine**; and
> 4. A **GPU Cross-Encoder Reranker** that calculates joint query-passage attention. Let us examine how this functions in detail."

---

### SLIDE 14: Modular RAG Mechanics & Naive RAG Failures
* **Slide Title:** *Modular RAG Mechanics & Naive RAG Failures*
* **Cards Displayed:** Left Card (*NeuroAgent 4-Stage Modular RAG*) vs. Right Card (*Failure Modes of Naive RAG*)
* **⏱️ Estimated Time:** 90 seconds
* **👉 Visual Pointer Cues:** Point to the 4 stages on the left card. Then explain the two devastating clinical errors on the right card (Context Contamination and Temporal Blindness), ending on the quantitative impact statistic (cutting pre-critic drift from 14.6% to 4.2%).

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 14 details the engineering mechanics of our modular pipeline:
>
> **[Point to Left Card: 4-Stage Modular RAG]**  
> 1. **Stage 1 (Query Compiler):** The system does not search with a raw prompt. It compiles an optimized clinical query embedding exact physical biomarkers: for example, combining lesion volume ($80.9\text{ cm}^3$), ASPECTS score ($2/10$), onset time ($2.0\text{ hours}$), and patient symptoms into a structured medical query.
> 2. **Stage 2 (Dynamic Router):** Depending on the extracted lesion tier, the query is routed: if volume is $< 5\text{ cm}^3$, it routes to STRIVE-2 small vessel disease guidelines; if volume is $> 15\text{ cm}^3$, it routes to large vessel occlusion protocols; if onset is $> 4.5\text{ hours}$, it routes to DAWN/DEFUSE-3 extended endovascular guidelines.
> 3. **Stages 3 & 4 (Hybrid Search & Reranking):** Sparse BM25 and dense PubMedBERT retrieve an initial candidate pool of 15 chunks, which our GPU cross-encoder reranks down to the top-3 most clinically relevant passages with a Context Precision of 0.91.
>
> **[Point to Right Card: Failure Modes of Naive RAG]**  
> Why does Naive RAG fail without this?
> 1. **Context Contamination:** A flat vector search on a massive $80\text{ cm}^3$ stroke will frequently retrieve chunks discussing dual-antiplatelet therapy for tiny lacunar strokes simply because both documents share words like 'stroke' and 'ischemia'. The LLM blends the two, prescribing antiplatelets instead of emergency surgery!
> 2. **Temporal Blindness:** Naive embeddings cannot discern that a 2.0-hour presentation qualifies for IV thrombolysis while a 6.0-hour presentation strictly forbids it.
> 3. **Quantitative Impact:** Our 4-stage modular pipeline slashes pre-critic clinical hallucination drift from **14.6% down to 4.2%** before the Critic agent is even called."

#### 🔬 Plain-English Concept Explanations:
* **Biological / Medical Aspect (Small Lacunar Stroke vs. Massive Territorial Stroke):**
  * Treating all strokes the same is like treating a paper cut the same as an amputated limb.
  * A **lacunar stroke** ($<5\text{ cm}^3$) is a blockage of a tiny microscopic deep penetrating artery in the brain. It does not cause massive swelling and does not need invasive surgery; it needs blood pressure medication and aspirin (antiplatelet therapy).
  * A **large territorial stroke** ($>50\text{ cm}^3$) is a complete blockage of a major trunk artery feeding half the brain. It causes massive swelling, pushes the brain across the skull (herniation), and requires immediate emergency endovascular catheter surgery. If Naive RAG mixes up these two treatments, the patient will die.
* **Technical / Computational Aspect (What is Cross-Encoder Reranking?):**
  * **Bi-Encoder (Standard Vector Search):** Encodes the search question into a vector, encodes the document into a vector separately, and measures the angle (cosine distance) between them. It is fast, but it cannot compare individual words directly.
  * **Cross-Encoder:** Takes the question AND the document, glues them together, and passes them through a deep Transformer neural network simultaneously. Every single word in the question directly cross-examines every single word in the document via cross-attention. It is like having a human judge read both sentences side-by-side.

---

### SLIDE 15: Ablation 1 — Chunking Strategy & Overlap (13 PDFs)
* **Slide Title:** *Chunking Strategy & Overlap Ablation (13 Clinical Guidelines)*
* **Metrics Displayed:** `100%` Recall@3 (1200c/150 overlap) | `4,678` Chunks (vs. 14,199 in micro-chunking) | `0.8%` Boundary integrity (Fixed 500c) | `1.1%` Boundary integrity (Recursive)
* **⏱️ Estimated Time:** 60 seconds
* **👉 Visual Pointer Cues:** Point to the bar chart showing Title-Aware Chunking reaching 100% Recall@3. Contrast the 4,678 compact chunk count against the 14,199 bloated count of recursive splitting.

#### 🎙️ Word-for-Word Spoken Script:
> "On Slide 15, we present our **first empirical GPU ablation study**: evaluating chunking strategies across all 13 peer-reviewed clinical guidelines, executed on our RTX 4060 GPU.
>
> Many developers assume text chunking is a trivial hyperparameter. Our empirical data proves it is the single most critical determinator of retrieval success.
>
> We evaluated three distinct chunking paradigms:
> 1. Standard **Fixed Character Chunking** (500 characters, 0 overlap);
> 2. Standard **Recursive Character Splitting** (800 characters, 100 overlap); and
> 3. Our layout-aware **Unstructured Title-Aware Chunking** (1,200 characters, 150 overlap).
>
> As shown on the slide, Title-Aware Chunking achieves **100% Recall@3** across our stroke benchmark while generating a compact, highly efficient index of **4,678 chunks**.
>
> In stark contrast, Fixed Chunking achieved only **0.8% syntactic boundary integrity**, and Recursive Splitting achieved only **1.1% integrity**, while bloating the index to **14,199 chunks**. Let us look at why this happens."

---

### SLIDE 16: Title-Aware Chunking: Why It Wins
* **Slide Title:** *Title-Aware Chunking: Why It Wins*
* **Cards Displayed:** Left Card (*Unstructured chunk_by_title Mechanics*) vs. Right Card (*Why Fixed & Zero-Overlap Fail*)
* **⏱️ Estimated Time:** 75 seconds
* **👉 Visual Pointer Cues:** Point to the layout parsing elements on the left (Title, NarrativeText, ListItem, Table). Then point to the red alert on the right illustrating a severed clinical sentence.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 16 explains the mathematical and linguistic reasons behind Ablation 1:
>
> **[Point to Left Card: chunk_by_title Mechanics]**  
> Clinical guidelines are not flat novels; they are heavily structured legal and medical documents with nested section headers, bulleted inclusion criteria, tables, and footnotes.
> We deployed `PyMuPDF` and `Unstructured` layout-aware parsing. The parser classifies visual bounding boxes into semantic categories: `Title`, `NarrativeText`, `ListItem`, and `Table`.
> Our tuned hyperparameters set a maximum chunk size of 1,200 characters, trigger a new chunk after 800 characters, combine snippets under 300 characters, and maintain a 150-character overlap. This guarantees that subsection titles are permanently bound to their underlying medical rules.
>
> **[Point to Right Card: Why Fixed Chunking Fails]**  
> Why do fixed splitters fail so catastrophically in medicine?
> 1. **Boundary Severing:** Fixed-character splitters cut text blindly at character index 500. Over 98% of medical clauses are severed mid-sentence! A sentence reading: *'Intravenous alteplase is strictly contraindicated if the patient has taken oral anticoagulants within 48 hours'* gets split directly between *'contraindicated if'* and *'patient has taken'*. The retrieval engine retrieves the first half and concludes alteplase is safe!
> 2. **Loss of Contextual Anchors:** With zero overlap, section headers like *'Exclusion Criteria for Reperfusion'* are isolated into their own chunk, leaving the underlying bullet points orphaned with zero context.
> 3. **BERT Token Alignment:** 1,200 characters corresponds to approximately 260 words, fitting perfectly inside the 512-token context window of transformer embedding models without truncating critical clinical criteria."

---

### SLIDE 17: Biomedical vs. General Embeddings & HNSW
* **Slide Title:** *Biomedical vs. General Embeddings & HNSW Indexing*
* **Metrics Displayed:** `100%` PubMedBERT Recall@3 | `0.8333` PubMedBERT MRR | `87.5%` MiniLM Recall@3 | `87.5%` BGE-small Recall@3
* **⏱️ Estimated Time:** 60 seconds
* **👉 Visual Pointer Cues:** Direct attention to the comparison table and bar chart: highlight PubMedBERT's superior 0.8333 Mean Reciprocal Rank (MRR) compared to MiniLM (0.7917) and BGE-small (0.7500).

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 17 presents our **second empirical GPU ablation study**: benchmarking domain-specific biomedical embeddings against general-domain embedding models across all 4,836 ChromaDB chunks on our RTX 4060 GPU.
>
> We re-encoded our entire stroke knowledge base and queried across three distinct transformer architectures:
> 1. Domain-specific **NeuML/pubmedbert-base-embeddings** (768 dimensions);
> 2. General-domain **all-MiniLM-L6-v2** (384 dimensions); and
> 3. High-ranking general-domain **BAAI/bge-small-en-v1.5** (384 dimensions).
>
> As shown on the slide, **PubMedBERT achieves 100.0% Recall@3** and the highest **Mean Reciprocal Rank of 0.8333**, outperforming both MiniLM and BGE-small on domain-specific acute stroke queries."

---

### SLIDE 18: GPU Embedding Benchmark & HNSW Indexing
* **Slide Title:** *GPU Embedding Benchmark & HNSW Indexing*
* **Cards Displayed:** Left Card (*Benchmark: 4,836 Chunks on CUDA*) vs. Right Card (*ChromaDB HNSW Graph Index*)
* **⏱️ Estimated Time:** 75 seconds
* **👉 Visual Pointer Cues:** Point to the latency metric on the left card (7.73 ms on RTX 4060). Then point to the HNSW graph topology parameters ($M=16, efConstruction=200$) and cosine distance formula on the right card.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 18 explains why biomedical pre-training wins and how our vector graph index operates:
>
> **[Point to Left Card: GPU Benchmark]**  
> * On our RTX 4060 GPU, PubMedBERT executes dense vector encoding and query matching in only **7.73 milliseconds per query**—faster than MiniLM at 11.03 ms.
> * Why does PubMedBERT beat BGE-small in clinical accuracy? General models are trained on Wikipedia, news articles, and web forums. When they encounter medical acronyms like **'ASPECTS'**, **'rt-PA'**, or **'STRIVE-2'**, their sub-word tokenizers shatter them into meaningless character fragments (`AS`, `PEC`, `TS`). PubMedBERT was pre-trained on millions of peer-reviewed biomedical PubMed articles; its vocabulary recognizes stroke nomenclature as distinct, unified semantic concepts.
>
> **[Point to Right Card: HNSW Graph Indexing]**  
> * In ChromaDB, we construct a Hierarchical Navigable Small World graph index with $M = 16$ bi-directional connections per node and an exploration factor of $efConstruction = 200$. This enables nearest-neighbor traversal in $O(\log N)$ logarithmic time.
> * Similarity is computed via normalized cosine distance:
> $$D_{\text{cosine}}(u, v) = 1 - \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
> * Top clinical guideline matches consistently achieve tight cosine distances of 0.22 to 0.28.
> * Crucially, because all 4,836 embeddings and ChromaDB operate 100% locally on-device, **zero Protected Health Information (PHI) ever leaves the hospital**, ensuring complete HIPAA and DPDP regulatory compliance."

---

### SLIDE 19: Retrieval Pipeline: BM25 vs. Dense vs. Reranker
* **Slide Title:** *Retrieval Pipeline: BM25 vs. Dense vs. Reranker*
* **Metrics Displayed:** `0.8125` BM25 MRR | `0.8333` Dense PubMedBERT MRR | `0.8542` Hybrid RRF MRR | `0.9167` RRF + Cross-Encoder MRR
* **⏱️ Estimated Time:** 60 seconds
* **👉 Visual Pointer Cues:** Point to the upward stair-step progression across the 4 stages: showing MRR climbing monotonically from 0.8125 to 0.8333 to 0.8542, and peaking at 0.9167 with the Cross-Encoder.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 19 presents our **third empirical GPU ablation study**: evaluating our multi-stage retrieval pipeline progression on the RTX 4060 GPU.
>
> We compared four retrieval configurations across our validated stroke question-answering benchmark:
> 1. Standalone **Sparse BM25** keyword search;
> 2. Standalone **Dense PubMedBERT** vector search;
> 3. **Hybrid Reciprocal Rank Fusion (RRF)** combining both; and
> 4. **Hybrid RRF plus a GPU Cross-Encoder Reranker**.
>
> The empirical results demonstrate a clean, monotonic performance gain at every single engineering stage:
> * Sparse BM25 achieves an MRR of **0.8125**.
> * Dense PubMedBERT improves this to **0.8333**.
> * Hybrid RRF merges their rankings, climbing to **0.8542**.
> * Finally, adding our GPU Cross-Encoder pushes Mean Reciprocal Rank to **0.9167**, achieving **87.5% Recall@1** and **100.0% Recall@3** in just **51.69 milliseconds**. Let us examine why this hybrid synergy is essential."

---

### SLIDE 20: Hybrid Search & Cross-Encoder Reranking
* **Slide Title:** *Hybrid Search & Cross-Encoder Reranking*
* **Cards Displayed:** Left Card (*Stage 1: BM25 + Dense RRF*) vs. Right Card (*Stage 2: Cross-Encoder Rerank*)
* **⏱️ Estimated Time:** 75 seconds
* **👉 Visual Pointer Cues:** Point to the RRF formula on the left ($RRF(d) = \sum \frac{1}{k + r(d)}$ with $k=60$). Then point to the Transformer joint attention diagram on the right (`[CLS] Query [SEP] Chunk`).

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 20 explains why hybrid search and cross-encoders are biologically and computationally necessary in medicine:
>
> **[Point to Left Card: Stage 1 Hybrid RRF]**  
> * Medical queries have a dual nature: they contain exact numerical drug dosages ('0.9 mg/kg', 'ASPECTS 6') which require lexical precision, and complex conceptual symptoms ('facial droop', 'wake-up stroke') which require semantic understanding.
> * Sparse BM25 excels at finding exact keywords like '0.9 mg/kg', but fails on synonyms. Dense PubMedBERT understands medical concepts, but struggles with exact numerical thresholds.
> * We merge both rankings using **Reciprocal Rank Fusion (RRF)** with smoothing constant $k = 60$:
> $$RRF\_Score(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{60 + r_m(d)}$$
> This generates a diverse, high-recall candidate pool of the top 15 candidate passages.
>
> **[Point to Right Card: Stage 2 Cross-Encoder]**  
> * In Stage 2, our GPU passes the candidate pool through `cross-encoder/ms-marco-MiniLM-L-6-v2`.
> * Unlike bi-encoders that encode query and document in isolation, the cross-encoder concatenates both into a single sequence: `[CLS] Query [SEP] Passage [EOS]`. Every layer of self-attention directly compares every query word against every passage word.
> * This achieves an extraordinary **82.5% clinical concept recall** across complex multi-guideline questions in only 51 milliseconds, completely eliminating irrelevant candidate chunks."
"""

