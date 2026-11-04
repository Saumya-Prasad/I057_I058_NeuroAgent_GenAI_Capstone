# script_generator/part1.py

def get_part1_text():
    return """# NeuroAgent: Comprehensive 40-Slide Master's Capstone Defense Script & Clinical-Technical Manual
**Companion Presentation Deck:** [NeuroAgent_Master_Capstone_Deck_Redesigned.pptx](file:///c:/Users/Saumya_Shambhavi/Downloads/GEN%20AI%20+%20CAPSTONE/NeuroAgent_Master_Capstone_Deck_Redesigned.pptx)  
**Authors:** Saumya Prasad & Shambhavi Prasad | **Course:** Generative AI & Master's Capstone (2026)  
**Hardware Verification:** NVIDIA GeForce RTX 4060 Laptop GPU (`cuda:0`, 8 GB VRAM)  
**Evaluation Cohort:** ATLAS v3.0 ($n=1,453$), BIOSE Hospital Report Corpus ($n=34$), Stroke QA Benchmark ($n=50$)

> [!IMPORTANT]
> **How to Use This Comprehensive Defense Script:**
> Every single slide from Slide 1 to Slide 40 contains:
> 1. **Slide Purpose & Pacing:** Exact target time to keep the complete defense within 20–25 minutes.
> 2. **Visual Pointer Cues:** What exact diagrams, numbers, or cards on the slide to direct the examiner's eyes to.
> 3. **Word-for-Word Spoken Script:** Authoritative, publication-grade academic delivery with zero filler words.
> 4. **Explanations for Anyone (Plain English & Layman Concepts):**
>    - **Biological & Medical Aspects:** What actually happens in the human brain, neurons, blood vessels, and why doctors care.
>    - **Technical & Computational Aspects:** How the algorithms, neural networks, equations, and code actually work under the hood.
> 5. **Viva Defense Pro-Tip:** The exact trap question examiners love to ask on this topic and your bulletproof response.

---

## PART 1: OVERVIEW, CLINICAL PROBLEM & SYSTEM ARCHITECTURE (SLIDES 1 – 6)

---

### SLIDE 1: Title Slide — Multimodal Medical AI & Agentic RAG
* **Slide Title:** *NeuroAgent: 3D Volumetric Perception, Domain-Specific Modular RAG & LangGraph Multi-Agent Architecture for Stroke Neuro-Radiology*
* **Metrics Displayed:** `309k` ACS-UNet parameters | `100%` Recall@3 (PubMedBERT) | `0.0%` Final clinical drift
* **⏱️ Estimated Time:** 60 seconds
* **🎯 Speaker Goal:** Establish the interdisciplinary bridge connecting your accepted 3D computer vision work with clinical Generative AI, while anchoring the presentation on four empirical pillars.

#### 🎙️ Word-for-Word Spoken Script:
> "Respected members of the examination committee and distinguished faculty, good morning. Today, Shambhavi and I are honored to present our Master's Capstone project: **NeuroAgent: 3D Volumetric Perception, Domain-Specific Modular RAG, and LangGraph Multi-Agent Architecture for Stroke Neuro-Radiology**.
>
> In clinical healthcare today, computer vision and Generative AI are often treated as completely separate worlds. Computer vision models output raw numbers or colored 3D heatmaps, but in an emergency room, a colored pixel map cannot treat a stroke patient. On the other hand, commercial Large Language Models produce fluent text, but they hallucinate critical numbers, alter dosages, and cannot see 3D volumetric MRI scans.
>
> **NeuroAgent** is an autonomous, neuro-symbolic clinical system that bridges this entire divide. It takes a raw 3D patient brain MRI, segments the acute ischemic lesion with a lightweight 309,000-parameter vision model, extracts exact mathematical biomarkers, queries 13 peer-reviewed stroke guidelines via a 4-stage Modular RAG pipeline, and synthesizes a certified American College of Radiology report through a closed-loop LangGraph multi-agent architecture.
>
> Every single component presented today has been evaluated through 5 live empirical ablation studies executed on our NVIDIA GeForce RTX 4060 GPU with zero simulated data. Let us begin with an overview of our architecture."

#### 🔬 Plain-English Concept Explanations:
* **Biological / Medical Aspect (What is an Ischemic Stroke?):**
  * The human brain requires a constant, uninterrupted supply of oxygen and glucose carried by blood. An **ischemic stroke** occurs when a blood clot (thrombus or embolus) lodges in a cerebral artery—most commonly the Middle Cerebral Artery (MCA)—blocking blood flow. Within minutes, brain cells deprived of oxygen lose their energy supply (ATP), their cell membranes collapse, water rushes into the cells (causing **cytotoxic edema**), and brain cells begin to die. This is why neurologists say *"Time is Brain"*: an untreated stroke destroys approximately **1.9 million neurons every single minute**.
* **Technical / Computational Aspect (What is Multimodal Medical AI & Agentic RAG?):**
  * **Multimodal** means the AI processes more than one type of data: it reads both **3D imaging tensors** (spatial voxel arrays from MRI scanners) and **unstructured text** (clinical patient histories and medical guideline PDFs).
  * **Agentic RAG** means the retrieval is not a dumb keyword search. Instead, autonomous software "agents" act like digital specialists: one calculates lesion measurements, one searches the medical literature, one writes the draft report, and an independent critic checks for medical errors before certifying the document.

#### 💡 Viva Defense Pro-Tip:
* *Examiner Question:* "Why did you build an end-to-end system instead of just training a better segmentation model or fine-tuning an LLM?"
* *Your Answer:* "Because in acute stroke triage, segmentation alone leaves clinicians with an uninterpreted tensor, while fine-tuning an LLM causes catastrophic forgetting and hallucinated drug doses. An end-to-end neuro-symbolic pipeline grounds the generative language model in exact physical 3D measurements and auditable clinical guidelines."

---

### SLIDE 2: NeuroAgent at a Glance
* **Slide Title:** *NeuroAgent at a Glance — 4 Foundational Pillars*
* **Cards Displayed:** 3D Perception Layer | Modular RAG Pipeline | LangGraph Multi-Agent System | Quantitative GPU Ablations
* **⏱️ Estimated Time:** 45 seconds
* **👉 Visual Pointer Cues:** Sweep left to right across the 4 color-coded cards, briefly naming the specific parameter count, embedding dimension, agent framework, and GPU hardware.

#### 🎙️ Word-for-Word Spoken Script:
> "To understand how NeuroAgent works, consider our four engineering pillars shown here:
>
> 1. **First, the 3D Perception Layer:** We deploy ACS-UNet, our 3D convolutional neural network with only 309,387 parameters. Evaluated on the 1,453-scan ATLAS v3.0 benchmark, it achieves a 61.91% 5-fold mean Dice score in just 146 milliseconds while running in only 4.2 GB of VRAM.
> 2. **Second, the Modular RAG Pipeline:** We reject naive vector search. We implement domain-specific PubMedBERT 768-dimensional embeddings and ChromaDB HNSW indexing across 13 peer-reviewed stroke guidelines and 34 hospital reports, delivering 100% Recall@3 in under 15 milliseconds.
> 3. **Third, the LangGraph Multi-Agent Architecture:** We structure clinical reasoning as a cyclical state machine. A local, on-premise Llama-3-8B model drafts the clinical report, while an independent GPT-4o-mini Critic agent equipped with 5 deterministic Python AST safety gates audits every sentence.
> 4. **Fourth, Quantitative GPU Ablations:** Every architectural claim is proven through 5 rigorous ablations run on an NVIDIA GeForce RTX 4060 Laptop GPU, verifying chunking, embeddings, hybrid retrieval, context window size, and causal decoding temperatures."

#### 🔬 Plain-English Concept Explanations:
* **Biological / Medical Aspect (Why Triage Needs Exact Biomarkers):**
  * When an emergency doctor decides whether to administer clot-busting medication, they cannot guess. They need to know the exact physical volume of dead tissue in cubic centimeters ($\text{cm}^3$). If the dead core is too big, restoring blood flow will cause the fragile, dead blood vessels to burst, causing a massive, fatal brain bleed.
* **Technical / Computational Aspect (What is an HNSW Index and AST?):**
  * **HNSW (Hierarchical Navigable Small World):** A smart graph-based indexing algorithm. Imagine a vector database as a multi-floor library: HNSW allows the AI to skip across top-floor express highways to find the exact nearest book in milliseconds without reading every shelf.
  * **AST (Abstract Syntax Tree):** A deterministic computer science tool that parses computer code or text into a tree of structural tokens. Instead of trusting an AI's feelings, an AST strictly inspects the text like a compiler, ensuring numbers match reality to the decimal point.

---

### SLIDE 3: Presentation Roadmap
* **Slide Title:** *Presentation Roadmap — Five Architectural Parts*
* **Roadmap Items:** 1. Architecture & Problem | 2. 3D Perception Layer | 3. Modular RAG | 4. LangGraph Multi-Agent | 5. Output & Evaluation
* **⏱️ Estimated Time:** 30 seconds
* **👉 Visual Pointer Cues:** Point to the sequence numbers 1 through 5, highlighting that the talk moves logically from clinical necessity to computer vision, retrieval, multi-agent control, and real empirical GPU proof.

#### 🎙️ Word-for-Word Spoken Script:
> "Our presentation is organized into five logical sections:
> * In **Part 1**, we examine the clinical emergency bottlenecks and why standard Generative AI fails in medicine.
> * In **Part 2**, we explore our 3D ACS-UNet perception model, explaining how depthwise separable convolutions and contralateral hemispheric symmetry enable edge deployment.
> * In **Part 3**, we present our Modular RAG framework and our first three empirical ablations on chunking, biomedical embeddings, and hybrid reranking.
> * In **Part 4**, we unpack our LangGraph cyclical state machine, dual-LLM routing, and the 5-Gate Critic self-correction loop.
> * Finally, in **Part 5**, we examine the certified clinical report output, benchmark evaluation across ATLAS and BIOSE, and our final two ablations on context window size and decoding temperatures."

---

### SLIDE 4: Clinical Bottlenecks & LLM Failure Modes
* **Slide Title:** *Clinical Bottlenecks & LLM Failure Modes*
* **Cards Displayed:** Left Red Card (*Clinical & 3D Vision Bottlenecks*) vs. Right Blue Card (*Generative AI & Safety Failure Modes*)
* **⏱️ Estimated Time:** 90 seconds
* **👉 Visual Pointer Cues:** First point to the 4 bullet points on the left card (Manual planimetry, Lacunar ambiguity, Hospital VRAM ceiling, Semantic gap). Then point across to the 4 failure modes on the right card (Numerical token drift, Catastrophic triage, Guideline staleness, Zero provenance).

#### 🎙️ Word-for-Word Spoken Script:
> "To understand why NeuroAgent is necessary, we must examine the severe failure modes existing in both clinical radiology and modern Generative AI.
>
> **[Point to Left Card: Clinical Bottlenecks]**  
> In hospital stroke care:
> 1. **Manual Planimetry Latency:** Clinicians still estimate stroke volume using primitive geometric approximations like ABC/2 or slice-by-slice manual contouring. This takes 15 to 25 minutes per patient—a catastrophic delay when 1.9 million neurons die every minute inside the hyperacute 4.5-hour window.
> 2. **Sub-5 cm³ Lacunar Ambiguity:** Small subcortical infarcts smaller than 5 cubic centimeters account for 25% of all strokes, yet on standard T1-weighted MRI they are nearly indistinguishable from benign age-related enlarged perivascular spaces.
> 3. **The Hospital VRAM Ceiling:** Existing deep learning baselines—such as nnU-Net with 31 million parameters and Swin-UNETR with 62 million parameters—require 14 to 18 Gigabytes of GPU VRAM. Standard hospital diagnostic workstations only have 6 to 8 Gigabyte consumer GPUs, leading to immediate Out-Of-Memory system crashes.
> 4. **The Semantic Gap:** Even when a vision model runs, it outputs a raw binary NIfTI array. Clinicians do not have time to parse a tensor; they need a structured American College of Radiology report.
>
> **[Point to Right Card: Generative AI Safety Failure Modes]**  
> Conversely, why can't we simply plug in ChatGPT or an off-the-shelf LLM?
> 1. **Numerical Token Drift:** Autoregressive language models mutate numbers. An infarct of 80.924 cubic centimeters gets rewritten as 8.09 cubic centimeters because LLMs predict word tokens rather than understanding physical volume.
> 2. **Catastrophic Triage Errors:** If an LLM recommends intravenous thrombolysis to a patient whose stroke core exceeds 50 cubic centimeters or whose onset exceeds 4.5 hours, the patient faces fatal hemorrhagic transformation.
> 3. **Guideline Staleness:** Static LLM weights lack the 2026 AHA/ASA guidelines, STRIVE-2 small vessel disease criteria, and ESO 2024 updates.
> 4. **Zero Provenance:** Standard LLMs write fluent, plausible-sounding medical prose that lacks bracketed, verifiable citations back to medical literature, making peer review impossible."

#### 🔬 Plain-English Concept Explanations:
* **Biological / Medical Aspect (Hemorrhagic Transformation & ABC/2):**
  * **ABC/2 Formula:** A crude clinical formula where a doctor measures the maximum length ($A$), width ($B$), and number of slices ($C$) of a stroke on a CT/MRI scan, assuming the stroke is a perfect sphere or ellipsoid. In reality, strokes have complex, irregular, branching shapes.
  * **Hemorrhagic Transformation:** When brain tissue dies from ischemia, the tiny capillaries inside the dead zone also die and their structural walls dissolve. If a powerful clot-dissolving drug (IV-rtPA) is administered too late (after 4.5 hours) or into a massive dead zone ($>50\text{ cm}^3$), the suddenly restored blood pressure bursts the dead capillary walls, causing catastrophic bleeding directly into the brain tissue.
* **Technical / Computational Aspect (What is Numerical Token Drift & NIfTI?):**
  * **Token Drift:** LLMs do not see numbers as whole mathematical quantities. The number `80.924` is broken into tokens: `80`, `.`, and `924`. During text generation, sampling randomness can easily cause the model to output `8.09`, accidentally shrinking the reported stroke size by a factor of 10!
  * **NIfTI Format (`.nii.gz`):** The standard medical file format for neuroimaging. Unlike a 2D JPEG photo, a NIfTI file is a 3-dimensional grid of voxels containing spatial header information (voxel spacing in millimeters, spatial orientation, and coordinate matrices).

---

### SLIDE 5: End-to-End Neuro-Symbolic & Agentic System (Architecture Diagram)
* **Slide Title:** *End-to-End Neuro-Symbolic & Agentic System*
* **Metrics Displayed:** `309k` ACS-UNet params | `768-d` PubMedBERT embeddings | `<15 ms` top-3 retrieval | `≤2` critic iterations
* **⏱️ Estimated Time:** 60 seconds
* **👉 Visual Pointer Cues:** Point to the overarching pipeline: input NIfTI and free-text history entering on the left, passing through the 3D Perception and Modular RAG engines, entering the LangGraph state machine, and exiting as a certified clinical report on the right.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 5 illustrates our complete end-to-end neuro-symbolic architecture.
>
> On the left, our system accepts two raw hospital inputs: the patient's 3D T1-weighted structural MRI scan in NIfTI format, and the emergency clinician's free-text clinical history notes.
>
> The processing flows through two coordinated subsystems:
> * **The Neural Perception Subsystem:** Our 309,000-parameter ACS-UNet segments the 3D scan in 146 milliseconds. A deterministic tensor module extracts exact physical biomarkers: volume in cubic centimeters, voxel count, laterality, and ASPECTS score. Simultaneously, our Clinical History Parser extracts onset hours and contraindication flags.
> * **The Symbolic & Knowledge Subsystem:** These exact biomarkers dynamically formulate a targeted clinical query. Our 4-stage Modular RAG pipeline searches ChromaDB across 13 stroke guidelines using PubMedBERT embeddings, retrieving relevant guideline passages in under 15 milliseconds.
> * **The LangGraph Agentic Loop:** The biomarkers and retrieved passages enter a typed state machine. Our on-premise Llama-3-8B Drafter synthesizes the report, which is audited by our GPT-4o-mini Critic. If any safety invariant fails, the state machine routes the report back for revision, converging in 2 or fewer iterations to produce a certified, auditable clinical document."

#### 🔬 Plain-English Concept Explanations:
* **Biological / Medical Aspect (Why Both Scan and History Are Needed):**
  * An MRI scan alone cannot tell you *when* the stroke started. A patient could have a small $10\text{ cm}^3$ stroke, but if it happened 3 days ago, clot-busting medication is useless and dangerous. Conversely, knowing a patient has arm weakness does not reveal if they have a small vessel blockage or a massive arterial occlusion. Safe medical decisions require marrying physical anatomy from the scan with temporal history from the patient.
* **Technical / Computational Aspect (What Does 'Neuro-Symbolic' Mean?):**
  * **Neuro:** Refers to deep neural networks (ACS-UNet, Transformers, LLMs). Neural nets excel at fuzzy perception, pattern recognition, and fluent writing, but they are prone to hallucinations.
  * **Symbolic:** Refers to classical, deterministic rules (boolean logic, regular expressions, mathematical constraints). Symbolic systems cannot hallucinate: $2 + 2$ is always $4$.
  * **Neuro-Symbolic AI:** The fusion of both. Neural networks do the heavy lifting of image perception and text drafting, while symbolic rules lock down the numbers, clinical thresholds, and safety gates.

---

### SLIDE 6: Four-Stage System Walkthrough
* **Slide Title:** *Four-Stage System Walkthrough*
* **Sections Displayed:** Stages 1 & 2: Perception & RAG | Stages 3 & 4: LangGraph & 5-Gate Critic
* **⏱️ Estimated Time:** 60 seconds
* **👉 Visual Pointer Cues:** Walk through the four numbered stages in order: 1 (Segmentation), 2 (History Parsing & RAG), 3 (LangGraph Orchestration), 4 (Critic Audit & Certification).

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 6 provides a step-by-step walkthrough of how data transforms through our four stages:
>
> * **In Stage 1 (Perception):** The patient's 3D NIfTI scan is normalized and fed into ACS-UNet. The model outputs a voxel-level segmentation mask, which is converted into physical volume ($V_{\text{cm}^3}$), hemispheric asymmetry ($\Delta_{\text{sym}}$), and an automated 10-point ASPECTS score.
> * **In Stage 2 (Knowledge Retrieval):** Our Clinical History Parser extracts the exact symptom onset duration ($t_{\text{onset}}$) and scans for contraindications such as active anticoagulation. These parameters compile a dynamic clinical query, and our hybrid retrieval engine fetches the top-3 guideline chunks from ChromaDB.
> * **In Stage 3 (LangGraph Synthesis):** The state machine loads all biomarkers and guideline evidence into a typed `AgentState`. Our local Llama-3-8B Drafter agent writes a complete American College of Radiology structured report, embedding bracketed numbered citations `[1]` through `[6]` for every statement.
> * **In Stage 4 (Verification & Certification):** Our independent Critic Agent runs 5 deterministic AST and regex verification gates. If any number, laterality, or guideline rule is violated, the report is routed back to the Drafter with an explicit error trace. Once all gates pass, the report is certified for the medical record."
"""

