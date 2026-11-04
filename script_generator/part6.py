# script_generator/part6.py

def get_part6_text():
    return """
---

## PART 6: CONCLUSION, FUTURE HORIZONS & VIVA DEFENSE GUIDE (SLIDES 38 – 40)

---

### SLIDE 38: Core Technical Contributions
* **Slide Title:** *Core Technical Contributions — Four Engineering Breakthroughs*
* **Cards Displayed:** 1. Edge-Deployable 3D Perception | 2. Biomedical Modular RAG | 3. LangGraph Cyclic Multi-Agent | 4. Neuro-Symbolic Safety
* **⏱️ Estimated Time:** 75 seconds
* **👉 Visual Pointer Cues:** Sweep across the 4 numbered contribution cards, summarizing the standout engineering metric for each pillar.

#### 🎙️ Word-for-Word Spoken Script:
> "In conclusion, Slide 38 summarizes the four core technical contributions of our Capstone project:
>
> 1. **Edge-Deployable 3D Perception:** We proved that a 309,000-parameter architecture using depthwise separable convolutions and contralateral hemispheric symmetry achieves a 61.91% 5-fold mean Dice score in 146 milliseconds while running in only 4.2 GB of VRAM, outperforming 30-million-parameter baselines on small subcortical lesions.
> 2. **Biomedical Modular RAG:** We proved that title-aware layout parsing, PubMedBERT 768-dimensional embeddings, and cross-encoder reranking achieve 100% Recall@3 and a 0.9167 Mean Reciprocal Rank across 13 stroke guidelines, operating 100% locally with zero patient privacy egress.
> 3. **LangGraph Cyclic Multi-Agent Architecture:** We demonstrated that splitting clinical reasoning into an adversarial state machine—pairing an on-premise Llama-3-8B Drafter with an independent Critic—eliminates self-preference bias and enables automated self-correction.
> 4. **Neuro-Symbolic Safety Bridge:** By binding the generative language model with deterministic 3D voxel integration and 5 AST verification gates, we eliminated numerical drift, achieved a 0.96 RAGAS Faithfulness score, and proved 0.0% final clinical drift."

---

### SLIDE 39: Future Research Horizons
* **Slide Title:** *Future Research Horizons*
* **Cards Displayed:** 1. Multiparametric Acute MRI | 2. Domain-Adapted QLoRA ($r=16$) | 3. 3D Counterfactual Inpainting | 4. DICOMweb & HL7 FHIR
* **⏱️ Estimated Time:** 60 seconds
* **👉 Visual Pointer Cues:** Walk through the 4 numbered forward-looking engineering milestones.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 39 outlines our immediate roadmap for clinical translation:
>
> 1. **Multiparametric Acute MRI:** Expanding our 3D vision encoder from structural T1-weighted scans to multi-sequence acute protocols: incorporating Diffusion-Weighted Imaging (DWI), Apparent Diffusion Coefficient (ADC), and FLAIR maps to automate diffusion-perfusion and DWI-FLAIR mismatch detection for wake-up strokes.
> 2. **Domain-Adapted QLoRA ($r=16$):** Fine-tuning local Llama-3-8B on over 10,000 de-identified hospital radiology reports from MIMIC-IV and BIOSE to refine professional phrasing while strictly preserving our non-parametric RAG retrieval for medical facts.
> 3. **3D Counterfactual Inpainting:** Replacing our simple left-right mirror flip with a 3D latent diffusion inpainting model that generates a photorealistic, synthetic 'pseudo-healthy' counterfactual brain of the patient to calculate subtle sub-voxel tissue loss.
> 4. **Hospital DICOMweb & HL7 FHIR Deployment:** Packaging NeuroAgent as an enterprise Docker microservice that connects directly to hospital Picture Archiving and Communication Systems (PACS), automatically triggering push notifications to emergency stroke teams the second an acute scan completes acquisition."

---

### SLIDE 40: Concluding Slide & Discussion
* **Slide Title:** *NeuroAgent: Safe, Auditable, Edge-Deployable Stroke Neuro-Radiology AI*
* **Metrics Displayed:** `61.91%` 5-Fold Dice | `100%` Recall@3 | `0.96` RAGAS Faithfulness | `0.0%` Clinical drift
* **⏱️ Estimated Time:** 30 seconds
* **👉 Visual Pointer Cues:** Gesture to the 4 final achievement numbers and invite questions from the panel.

#### 🎙️ Word-for-Word Spoken Script:
> "To conclude, NeuroAgent proves that clinical Generative AI does not require sacrificing mathematical rigor, patient privacy, or hardware accessibility.
>
> By uniting lightweight 3D volumetric vision, domain-specific Modular RAG, and an adversarial LangGraph multi-agent architecture, we have delivered a clinical system that is fast, safe, fully auditable, and ready for deployment on standard hospital workstations.
>
> Thank you for your time and attention. Shambhavi and I are now eager to take your questions."

---

## PART 7: COMPREHENSIVE MEDICAL & TECHNICAL GLOSSARY
*(Everything a non-specialist or examiner might ask about)*

### 🧠 Biological & Medical Terminology (Plain English):
1. **Ischemic Stroke:** A medical emergency where a blood clot blocks an artery in the brain, depriving neurons of oxygen and nutrients. Accounts for 87% of all strokes.
2. **Penumbra vs. Core:**
   - **Ischemic Core:** The central area of brain tissue that has already suffered irreversible necrosis (dead cells). It cannot be saved.
   - **Ischemic Penumbra:** The surrounding zone of stunned, starving brain tissue that is receiving minimal collateral blood flow. It is critically endangered but still salvageable if blood flow is restored quickly.
3. **Time is Brain:** A clinical aphorism reflecting the biological fact that an untreated acute ischemic stroke destroys approximately **1.9 million neurons and 14 billion synapses every single minute**.
4. **Intravenous Thrombolysis (IVT / rtPA / Alteplase / Tenecteplase):**
   - The injection of a powerful clot-dissolving drug into the bloodstream to break down fibrin clots.
   - **Window:** Must be administered within **4.5 hours** of symptom onset.
   - **Danger:** If administered to a patient with a massive stroke core ($>50\text{ cm}^3$) or after 4.5 hours, the fragile dead blood vessels burst, causing fatal **hemorrhagic transformation** (bleeding into the brain).
5. **Mechanical Thrombectomy (Endovascular Treatment / EVT):**
   - A minimally invasive catheter procedure where an interventional neuroradiologist threads a tiny wire through an artery from the groin or wrist up into the brain, deploying a tiny wire cage (stent retriever) or suction tube to physically pull the clot out.
   - **Window:** Effective up to 6 hours routinely, and up to **24 hours** in selected patients with salvageable penumbra (proven by the DAWN and DEFUSE-3 trials).
6. **ASPECTS (Alberta Stroke Programme Early CT Score):**
   - A 10-point quantitative anatomical scoring system evaluating 10 key Middle Cerebral Artery (MCA) brain regions.
   - Normal brain = $10/10$. Each dead region subtracts 1 point. A score of $\le 5$ indicates extensive, severe irreversible damage.
7. **Pyramidal Decussation:**
   - The biological crossing of motor nerve fibers in the caudal medulla at the base of the brainstem. Corticospinal fibers from the left side of the brain cross over to control the right side of the body, and vice versa.
8. **Lacunar Infarcts & Small Vessel Disease (SVD):**
   - Small strokes ($<5\text{ cm}^3$ or $<15\text{ mm}$ in diameter) occurring in the deep, microscopic penetrating arteries of the brain. They cause focal symptoms (like isolated motor weakness) but do not cause massive brain swelling.
9. **Wake-Up Stroke & DWI-FLAIR Mismatch:**
   - When a patient wakes up with stroke symptoms and the exact onset time is unknown.
   - **Diffusion-Weighted Imaging (DWI)** turns bright within 15–30 minutes of cell death (cytotoxic edema).
   - **FLAIR MRI** takes 4.5 hours to turn bright (vasogenic edema).
   - If DWI is bright and FLAIR is normal (mismatch), the stroke occurred less than 4.5 hours ago, making the patient eligible for clot-busting medication!

---

### 💻 Computer Science, AI & Mathematical Terminology:
1. **3D Voxel:** A three-dimensional pixel. While a pixel has width and height ($\Delta x, \Delta y$), a voxel also has depth ($\Delta z$), representing a physical $1\text{ mm}^3$ block of human brain tissue.
2. **Depthwise Separable 3D Convolution:**
   - A neural network optimization that splits a heavy $3 \times 3 \times 3$ convolution into a spatial-only depthwise convolution followed by a $1 \times 1 \times 1$ pointwise channel convolution, cutting mathematical operations by up to 90%.
3. **Dice Similarity Coefficient:**
   - A metric measuring the spatial overlap between two binary segmentations: $\text{Dice} = \frac{2 |A \cap B|}{|A| + |B|}$. A score of 1.0 indicates perfect overlap; 0.0 indicates zero overlap.
4. **VRAM (Video Random Access Memory):**
   - High-bandwidth physical memory soldered onto the GPU card. Models must fit entirely within VRAM to execute without crashing.
5. **RAG (Retrieval-Augmented Generation):**
   - An AI architecture that connects an LLM to an external searchable database of verified documents (like PDFs), forcing the LLM to base its answers on retrieved evidence rather than memorized weights.
6. **HNSW (Hierarchical Navigable Small World):**
   - A multi-layer graph algorithm used in vector databases to find the nearest matching semantic vectors in logarithmic $O(\log N)$ time.
7. **BM25:**
   - A probabilistic keyword-matching search algorithm (an advanced evolution of TF-IDF) that ranks documents based on the exact frequency of search terms.
8. **Cross-Encoder:**
   - A Transformer model that takes both the search query and the candidate document together as a single input, allowing every token in the query to attend directly to every token in the document via cross-attention.
9. **LangGraph State Machine:**
   - A software framework for building cyclic, stateful AI agent workflows using nodes (agents), edges (routing logic), and a centralized typed state object.
10. **LLM Temperature ($T$):**
    - A mathematical scaling parameter applied to the logits of an LLM before the softmax function. At $T = 0.0$ (greedy decoding), the model always chooses the highest-probability token, guaranteeing deterministic, reproducible outputs. Higher temperatures ($T > 0$) introduce randomness.
11. **RAGAS (Retrieval Augmented Generation Assessment):**
    - An automated evaluation framework measuring:
      - *Faithfulness:* Are the claims supported by retrieved context?
      - *Answer Relevance:* Does the answer address the question?
      - *Context Precision:* Are the relevant chunks ranked at the top?
      - *Context Recall:* Did the search retrieve all necessary facts?

---

## PART 8: TOP 10 TOUGH VIVA DEFENSE QUESTIONS & BULLETPROOF ANSWERS

1. **Examiner Question:** *"Your ACS-UNet achieves a 61.91% Dice score on ATLAS v3.0, whereas nnU-Net papers claim ~64-65%. Why should a hospital use your model?"*
   * **Your Answer:** "nnU-Net achieves ~64% Dice by using 31.2 million parameters and consuming 14.6 GB of VRAM, causing an immediate Out-Of-Memory crash on standard hospital workstations. ACS-UNet uses only 309,387 parameters—a 100-fold reduction—runs in 4.2 GB of VRAM, and executes in 146 milliseconds. Furthermore, on difficult small subcortical lesions (<5 cm³), our contralateral symmetry prior actually beats standard nnU-Net by +16.3% Dice."

2. **Examiner Question:** *"Why not just fine-tune Llama-3 on thousands of stroke radiology reports instead of using RAG?"*
   * **Your Answer:** "Fine-tuning updates parametric weights, which suffers from catastrophic forgetting and numerical drift. A fine-tuned model cannot cite the exact PDF page number, and when guidelines update (such as the 2026 AHA/ASA guideline), a fine-tuned model must be completely re-trained. RAG allows instant guideline updates in seconds and provides 100% auditable citations."

3. **Examiner Question:** *"Why did you use BM25 keyword search if dense vector embeddings are supposed to be more advanced?"*
   * **Your Answer:** "In medicine, exact numbers and acronyms matter. Dense embeddings understand conceptual meanings like 'clot' and 'thrombosis', but they struggle with exact numerical tokens like '0.9 mg/kg' or 'ASPECTS 6'. BM25 captures exact lexical matches, while PubMedBERT captures clinical semantics. Combining them via Reciprocal Rank Fusion gives us the best of both worlds, boosting MRR from 0.8125 to 0.8542."

4. **Examiner Question:** *"What prevents your LangGraph state machine from getting stuck in an infinite loop between Drafter and Critic?"*
   * **Your Answer:** "We enforce a hard mathematical bound: `max_reflections <= 2`. If a draft fails the Critic's gates twice, a deterministic Python AST fallback rule enforces the safety invariants directly, guaranteeing 100% finite termination."

5. **Examiner Question:** *"Why use two different LLMs (Llama-3-8B and GPT-4o-mini) instead of just using Llama-3 for both drafting and auditing?"*
   * **Your Answer:** "Using the same model creates self-preference bias: an LLM auditing its own draft will share the exact same blind spots and agree with its own errors. By pairing Meta's Llama-3 with an independent OpenAI GPT-4o model and deterministic Python AST rules, we create an adversarial cross-family check that drives clinical drift to 0.0%."

6. **Examiner Question:** *"How can you claim zero patient privacy leakage if you use GPT-4o-mini as a Critic?"*
   * **Your Answer:** "The Drafter runs locally on-premise, generating the report from raw MRI scans and identifiable clinical notes. By the time the draft reaches the Critic, all patient identifiers have been stripped; the Critic only audits anonymized medical statements against general guideline rules. Furthermore, the Critic can be swapped with a local quantized model with zero architectural changes."

7. **Examiner Question:** *"What happens if a patient has a bilateral stroke that affects both sides of the brain? Doesn't your contralateral symmetry assumption fail?"*
   * **Your Answer:** "Bilateral acute ischemic strokes represent less than 1.5% of all clinical stroke cases (usually basilar artery or embolic showers). Even in bilateral presentation, the lesions are almost never spatially symmetrical across the mid-sagittal plane, meaning the subtraction $|F(X) - \text{Flip}(F(X))|$ still generates strong non-zero asymmetric activation."

8. **Examiner Question:** *"Why is decoding temperature T = 0.0 mandatory in your pipeline?"*
   * **Your Answer:** "Our empirical Ablation 5 proved that any temperature above 0.0 introduces random probability sampling on floating-point tokens. At T = 0.7, numerical strings like 80.924 cm³ were mutated in 60% of runs. In medical AI, exact numerical fidelity is non-negotiable; greedy argmax decoding ($T=0.0$) guarantees 100% reproducible numerical fidelity."

9. **Examiner Question:** *"How did you handle the 4.5-hour time window in patients who wake up with stroke symptoms?"*
   * **Your Answer:** "Our Clinical History Parser explicitly detects wake-up and unwitnessed strokes. In accordance with the WAKE-UP trial protocol, standard time-based IV thrombolysis is flagged as contraindicated, and the system automatically prompts for an MRI DWI-FLAIR mismatch sequence to verify whether the tissue is under 4.5 hours old."

10. **Examiner Question:** *"What is the single most important metric proving your system is ready for clinical evaluation?"*
    * **Your Answer:** "Our 0.0% residual clinical drift across our 50-patient benchmark. While baseline Zero-Shot LLMs made safety violations in 28.4% of cases, our neuro-symbolic 5-gate critic caught and corrected every single violation, proving that AI can be bounded by deterministic clinical guardrails."
"""

