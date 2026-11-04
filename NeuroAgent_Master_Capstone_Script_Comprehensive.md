# NeuroAgent: Comprehensive 40-Slide Master's Capstone Defense Script & Clinical-Technical Manual
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
  * When an emergency doctor decides whether to administer clot-busting medication, they cannot guess. They need to know the exact physical volume of dead tissue in cubic centimeters ($	ext{cm}^3$). If the dead core is too big, restoring blood flow will cause the fragile, dead blood vessels to burst, causing a massive, fatal brain bleed.
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
  * **Hemorrhagic Transformation:** When brain tissue dies from ischemia, the tiny capillaries inside the dead zone also die and their structural walls dissolve. If a powerful clot-dissolving drug (IV-rtPA) is administered too late (after 4.5 hours) or into a massive dead zone ($>50	ext{ cm}^3$), the suddenly restored blood pressure bursts the dead capillary walls, causing catastrophic bleeding directly into the brain tissue.
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
  * An MRI scan alone cannot tell you *when* the stroke started. A patient could have a small $10	ext{ cm}^3$ stroke, but if it happened 3 days ago, clot-busting medication is useless and dangerous. Conversely, knowing a patient has arm weakness does not reveal if they have a small vessel blockage or a massive arterial occlusion. Safe medical decisions require marrying physical anatomy from the scan with temporal history from the patient.
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
> * **In Stage 1 (Perception):** The patient's 3D NIfTI scan is normalized and fed into ACS-UNet. The model outputs a voxel-level segmentation mask, which is converted into physical volume ($V_{	ext{cm}^3}$), hemispheric asymmetry ($\Delta_{	ext{sym}}$), and an automated 10-point ASPECTS score.
> * **In Stage 2 (Knowledge Retrieval):** Our Clinical History Parser extracts the exact symptom onset duration ($t_{	ext{onset}}$) and scans for contraindications such as active anticoagulation. These parameters compile a dynamic clinical query, and our hybrid retrieval engine fetches the top-3 guideline chunks from ChromaDB.
> * **In Stage 3 (LangGraph Synthesis):** The state machine loads all biomarkers and guideline evidence into a typed `AgentState`. Our local Llama-3-8B Drafter agent writes a complete American College of Radiology structured report, embedding bracketed numbered citations `[1]` through `[6]` for every statement.
> * **In Stage 4 (Verification & Certification):** Our independent Critic Agent runs 5 deterministic AST and regex verification gates. If any number, laterality, or guideline rule is violated, the report is routed back to the Drafter with an explicit error trace. Once all gates pass, the report is certified for the medical record."



---

## PART 2: 3D PERCEPTION LAYER & BIOMARKER EXTRACTION (SLIDES 7 – 10)

---

### SLIDE 7: ACS-UNet — Segmentation Results & Efficiency
* **Slide Title:** *ACS-UNet: Segmentation Results & Efficiency*
* **Metrics Displayed:** `309,387` parameters | `61.91%` 5-Fold Mean Dice | `146 ms` inference time | `4.2 GB` VRAM
* **⏱️ Estimated Time:** 60 seconds
* **👉 Visual Pointer Cues:** Point to the 3D volume rendering and coronal slice showing the ground truth red mask vs. the predicted green mask. Direct attention to the parameter count (309k) and the VRAM gauge (4.2 GB).

#### 🎙️ Word-for-Word Spoken Script:
> "In Part 2, we dive into our 3D vision foundation: **ACS-UNet**.
>
> On Slide 7, we present our model's core segmentation performance and resource footprint. ACS-UNet contains exactly **309,387 parameters**.
>
> When evaluated across the multi-center **ATLAS v3.0** cohort of 1,453 chronic and acute stroke patients using rigorous 5-fold cross-validation, ACS-UNet achieves a **61.91% 5-fold mean Dice similarity coefficient**.
>
> But what is truly game-changing for clinical deployment is our computational efficiency: the entire 3D volume inference executes in **146 milliseconds** while consuming only **4.2 Gigabytes of GPU VRAM**. This means our 3D perception engine runs comfortably on standard hospital workstation GPUs, such as our NVIDIA GeForce RTX 4060, completely bypassing the memory bottlenecks of existing academic baselines."

#### 🔬 Plain-English Concept Explanations:
* **Biological / Medical Aspect (What is a 3D MRI Voxel & What Does Dice Measure?):**
  * A 2D digital photo is made of flat squares called **pixels**. A 3D MRI brain scan is composed of tiny 3D cubes called **voxels** (volume elements). In our scans, each voxel represents exactly $1 	imes 1 	imes 1 	ext{ mm}^3$ of physical human brain tissue.
  * The **Dice Similarity Coefficient** measures how accurately the AI identifies the boundaries of the dead brain tissue. Imagine an expert neuroradiologist draws the exact boundaries of a stroke in red, and the AI draws its prediction in green. Dice measures the percentage of physical overlap between the two shapes. In 3D medical stroke imaging, lesions are extremely irregular, fuzzy, and variable, making a 61.91% Dice score on the challenging ATLAS v3.0 benchmark highly competitive with state-of-the-art models that are 100 times larger.
* **Technical / Computational Aspect (What is VRAM and Why Does 4.2 GB Matter?):**
  * **VRAM (Video Random Access Memory):** The ultra-fast memory located directly on the graphics card (GPU). When processing large 3D neural networks, the entire 3D image tensor and all intermediate neural activations must reside inside VRAM.
  * If a model requires 16 GB of VRAM and the hospital computer only has an 8 GB GPU, the computer throws a fatal `CUDA Out of Memory` error and halts completely. By requiring only 4.2 GB, ACS-UNet leaves half the GPU memory free to run our language model and retrieval engines simultaneously on the same machine!

---

### SLIDE 8: ACS-UNet — Attention, Symmetry & Efficiency
* **Slide Title:** *ACS-UNet: Attention, Symmetry & Efficiency*
* **Cards Displayed:** 1. Depthwise Separable 3D Convolutions | 2. Contralateral Symmetry ($\Delta_{	ext{sym}}$) | 3. Why Not nnU-Net / Swin-UNETR? | 4. Performance on ATLAS v3.0
* **⏱️ Estimated Time:** 90 seconds
* **👉 Visual Pointer Cues:** Walk through the 4 numbered cards: explain the mathematical factorization in Card 1, the anatomical mirror subtraction in Card 2, contrast against the memory hogs in Card 3, and summarize the small-lesion Dice boost in Card 4.

#### 🎙️ Word-for-Word Spoken Script:
> "How does a model with only 309,000 parameters compete with models containing tens of millions of weights? Slide 8 explains the two architectural breakthroughs inside ACS-UNet:
>
> 1. **Depthwise Separable 3D Convolutions:** Standard 3D convolutions simultaneously mix spatial coordinates ($X, Y, Z$) and channel features ($C$), leading to cubic parameter growth. We factorize each standard $3 	imes 3 	imes 3$ convolution into a depthwise spatial convolution followed by a $1 	imes 1 	imes 1$ pointwise channel projection. This slashes parameter count and computational FLOPs by over 90% without sacrificing spatial receptive fields.
> 2. **Contralateral Hemispheric Symmetry ($\Delta_{	ext{sym}}$):** The human brain exhibits natural bilateral symmetry across the mid-sagittal plane. Ischemic strokes, however, are almost exclusively unilateral. We exploit this biological prior by taking the encoder feature map $F_{	ext{enc}}(X)$, flipping it across the left-right axis, and computing the absolute feature difference: $|F_{	ext{enc}}(X) - 	ext{Flip}_{LR}(F_{	ext{enc}}(X))|$. This instantly illuminates acute tissue asymmetry, giving the network a massive inductive bias.
> 3. **Why NOT nnU-Net or Swin-UNETR?** Standard baselines like nnU-Net (31.2 million parameters) and Swin-UNETR (62.2 million parameters) are brute-force architectures requiring 14.6 to 18.4 GB of VRAM. More importantly, they treat brain MRI as generic 3D voxels, lacking any neurological symmetry priors.
> 4. **Empirical Impact on ATLAS v3.0:** On our 1,453-patient cohort, incorporating this contralateral symmetry prior yields a **+16.3% Dice improvement on difficult small subcortical lesions** under 5 cubic centimeters, while operating at 146 milliseconds."

#### 🔬 Plain-English Concept Explanations:
* **Biological / Medical Aspect (Bilateral Brain Symmetry & Unilateral Strokes):**
  * If you look down at a human brain, the left hemisphere and right hemisphere are anatomical mirror images of each other. While humans have minor functional differences (like language usually on the left), the physical layout of blood vessels and brain matter is symmetrical across the central mid-sagittal fissure.
  * When a stroke occurs, a blood clot almost never blocks both sides of the brain at once; it blocks an artery on the left OR on the right. By mathematically subtracting the left side of the brain from the right side, healthy brain structures cancel out to zero, while the fresh stroke lights up like a beacon!
* **Technical / Computational Aspect (Depthwise Separable 3D Convolutions):**
  * Imagine you are filtering coffee. A standard 3D convolution tries to grind the coffee beans and heat the water in a single massive, complex machine ($3 	imes 3 	imes 3 	imes C_{	ext{in}} 	imes C_{	ext{out}}$ multiplications).
  * A **depthwise separable convolution** splits the task into two simple steps: first, it filters each spatial slice independently (depthwise), and then it mixes the flavors using a simple $1 	imes 1 	imes 1$ operation (pointwise). This produces virtually the exact same mathematical output while requiring 100 times fewer internal weights!

---

### SLIDE 9: Volumetric Biomarker Extraction
* **Slide Title:** *Volumetric Biomarker Extraction*
* **Metrics Displayed:** `80.924 cm³` lesion volume (80,924 voxels) | `2 / 10` estimated ASPECTS | `0.0884` $\Delta_{	ext{sym}}$, Right MCA | `0.000 cm³` Healthy control
* **⏱️ Estimated Time:** 60 seconds
* **👉 Visual Pointer Cues:** Point to the red segmented stroke volume card ($80.924	ext{ cm}^3$), then to the low ASPECTS score ($2/10$), and contrast with the green Healthy Control card at the bottom displaying exactly $0.000	ext{ cm}^3$ and ASPECTS $10/10$.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 9 demonstrates how our perception engine bridges computer vision into clinical reality. Once ACS-UNet generates the voxel probability map, our deterministic pipeline extracts four crucial clinical biomarkers:
>
> * **First, Infarct Burden:** For this severe acute stroke patient, the segmented lesion measures **80.924 cubic centimeters**, corresponding to exactly **80,924 positive voxels** at $1	ext{ mm}^3$ resolution.
> * **Second, Topographic ASPECTS Score:** The model automatically maps the lesion against 10 critical anatomic Middle Cerebral Artery regions, estimating an ASPECTS score of **2 out of 10**.
> * **Third, Neuroanatomical Lateralization:** The model calculates a contralateral asymmetry index ($\Delta_{	ext{sym}}$) of **0.0884**, localizing the lesion unambiguously to the **Right Cerebral Hemisphere in the MCA territory**.
> * **Fourth, Healthy Control Verification:** Crucially, when we feed an entirely normal brain scan from a healthy control subject into our pipeline, the model outputs exactly **0.000 cubic centimeters** (0 voxels) and an ASPECTS score of **10 out of 10**, demonstrating **100% specificity** with zero false-positive hallucinations."

#### 🔬 Plain-English Concept Explanations:
* **Biological / Medical Aspect (What is ASPECTS and Why is 2/10 Alarming?):**
  * **ASPECTS (Alberta Stroke Programme Early CT Score):** A 10-point quantitative anatomical scoring system used worldwide by stroke neurologists. The brain's Middle Cerebral Artery (MCA) territory is divided into 10 key anatomical zones: 6 cortical areas (M1 through M6) and 4 deep subcortical structures (caudate nucleus, lentiform nucleus, internal capsule, and insular ribbon).
  * A normal, healthy brain receives a perfect score of **10/10**. For every zone that has died from stroke, 1 point is subtracted.
  * A score of **2/10** is an emergency alert: it means **8 out of the 10 vital brain regions are already irreversibly destroyed**! The patient has a massive territorial stroke and is at severe risk of malignant cerebral edema and brain herniation.
* **Technical / Computational Aspect (Why 100% Specificity on Healthy Controls Matters):**
  * Many commercial medical AI models suffer from "over-calling": they see tiny dark spots or scanner noise on a normal patient's MRI and falsely label it as an acute stroke. If an AI tells an emergency doctor that a healthy patient had a stroke, the patient might be given unnecessary, high-risk blood thinners. Proving our model outputs exactly `0.000 cm³` on healthy controls is essential for clinical trust.

---

### SLIDE 10: From Tensor to Biomarker to Safety Bridge
* **Slide Title:** *From Tensor to Biomarker to Safety Bridge*
* **Cards Displayed:** Left Card (*3D Tensor-to-Biomarker Extraction*) vs. Right Card (*Neuro-Symbolic Safety Bridge*)
* **⏱️ Estimated Time:** 75 seconds
* **👉 Visual Pointer Cues:** Point to the integration equation on the left card ($V_{	ext{cm}^3} = \sum I(P_{	ext{ens}} > 0.5) \dots$). Then trace across to the right card showing the boolean logic equation: $	ext{ivt\_eligible} = (0 < V < 50	ext{ cm}^3) \land (t_{	ext{onset}} \le 4.5	ext{h}) \land (
eg 	ext{Contraindicated})$.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 10 details our architectural safety bridge. How do we ensure that raw neural network predictions safely guide medical decisions without allowing a language model to invent numbers?
>
> **[Point to Left Card: Tensor-to-Biomarker Extraction]**  
> First, our pipeline calculates exact physical volume through discrete mathematical integration over the ensemble probability map $P_{	ext{ens}}$:
> $$V_{	ext{cm}^3} = \sum_{x,y,z} \mathbb{I}(P_{	ext{ens}}(x,y,z) > 0.5) 	imes rac{\Delta x \cdot \Delta y \cdot \Delta z}{1000}$$
> Here, each voxel's spatial dimensions ($1 	imes 1 	imes 1 	ext{ mm}^3$) are summed and converted to cubic centimeters. This gives us an exact, reproducible ground truth: 80.924 cubic centimeters.
>
> **[Point to Right Card: Neuro-Symbolic Safety Bridge]**  
> Second, our system constructs a hard deterministic safety bridge:
> 1. Our `ClinicalHistoryParser` applies deterministic regular expressions to extract symptom onset time ($t_{	ext{onset}}$) and flags clinical contraindications like active anticoagulant medications (e.g., Warfarin or Eliquis).
> 2. Then, before any Large Language Model is called, a pure Python function evaluates the official AHA/ASA and ESO reperfusion invariant:
> $$	ext{ivt\_eligible} = (0 < V < 50.0	ext{ cm}^3) \land (t_{	ext{onset}} \le 4.5	ext{ hours}) \land (
eg 	ext{HasClinicalContraindications})$$
> 3. These boolean decisions and exact numbers are stored in a structured JSON state object. The language model is **never permitted to perform mathematical calculations or make unconstrained medical decisions**. It receives verified facts as immutable symbolic constraints."

#### 🔬 Plain-English Concept Explanations:
* **Biological / Medical Aspect (Why the 50 cm³ and 4.5h Rules Exist):**
  * When an artery is blocked, brain cells die in two zones: the dead central **core** and the surrounding, struggling **penumbra**.
  * If the dead core exceeds **50 cubic centimeters**, the damage is so extensive that the blood vessel walls have begun to decompose. If a doctor injects intravenous clot-busting medication (IV-rtPA), the sudden rush of high-pressure blood will blow out the dead capillary walls, causing fatal bleeding inside the skull. The exact same risk occurs if more than **4.5 hours** have passed since symptoms began.
* **Technical / Computational Aspect (Why We Don't Let LLMs Do Math):**
  * Large Language Models are probabilistic text completion engines; they predict the next most likely token. When an LLM sees `80.924 > 50.0`, it does not perform a mathematical comparison on a CPU register; it simply guesses what words typically follow. Allowing an LLM to evaluate medical safety rules results in catastrophic failures. By evaluating the logic in pure Python code before the LLM is invoked, we achieve 100% mathematical determinism.



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
* **👉 Visual Pointer Cues:** Point to the mathematical marginalization formula on the left card ($P(y|x) pprox \prod \dots$). Then highlight the three failure modes on the right card: Catastrophic Forgetting, Dosage Drift, and Lack of Auditability.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 12 provides the formal mathematical and clinical justification for why RAG fundamentally outperforms parametric fine-tuning:
>
> **[Point to Left Card: Non-Parametric Generation]**  
> Formally, RAG models the probability of generating clinical token sequence $y$ given patient input $x$ by marginalizing over a set of top-$K$ retrieved evidence passages $z$:
> $$P(y|x) pprox \prod_{t=1}^T \sum_{z \in 	ext{Top-K}} P_{	ext{ret}}(z|x) \cdot P_	heta(y_t \mid x, z, y_{<t})$$
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
* **👉 Visual Pointer Cues:** Trace the top linear flow (Naive RAG: User Query $ightarrow$ Vector Search $ightarrow$ LLM) and contrast it with the bottom multi-stage pipeline (Compiler $ightarrow$ Router $ightarrow$ Hybrid RRF $ightarrow$ Cross-Encoder).

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
> 1. **Stage 1 (Query Compiler):** The system does not search with a raw prompt. It compiles an optimized clinical query embedding exact physical biomarkers: for example, combining lesion volume ($80.9	ext{ cm}^3$), ASPECTS score ($2/10$), onset time ($2.0	ext{ hours}$), and patient symptoms into a structured medical query.
> 2. **Stage 2 (Dynamic Router):** Depending on the extracted lesion tier, the query is routed: if volume is $< 5	ext{ cm}^3$, it routes to STRIVE-2 small vessel disease guidelines; if volume is $> 15	ext{ cm}^3$, it routes to large vessel occlusion protocols; if onset is $> 4.5	ext{ hours}$, it routes to DAWN/DEFUSE-3 extended endovascular guidelines.
> 3. **Stages 3 & 4 (Hybrid Search & Reranking):** Sparse BM25 and dense PubMedBERT retrieve an initial candidate pool of 15 chunks, which our GPU cross-encoder reranks down to the top-3 most clinically relevant passages with a Context Precision of 0.91.
>
> **[Point to Right Card: Failure Modes of Naive RAG]**  
> Why does Naive RAG fail without this?
> 1. **Context Contamination:** A flat vector search on a massive $80	ext{ cm}^3$ stroke will frequently retrieve chunks discussing dual-antiplatelet therapy for tiny lacunar strokes simply because both documents share words like 'stroke' and 'ischemia'. The LLM blends the two, prescribing antiplatelets instead of emergency surgery!
> 2. **Temporal Blindness:** Naive embeddings cannot discern that a 2.0-hour presentation qualifies for IV thrombolysis while a 6.0-hour presentation strictly forbids it.
> 3. **Quantitative Impact:** Our 4-stage modular pipeline slashes pre-critic clinical hallucination drift from **14.6% down to 4.2%** before the Critic agent is even called."

#### 🔬 Plain-English Concept Explanations:
* **Biological / Medical Aspect (Small Lacunar Stroke vs. Massive Territorial Stroke):**
  * Treating all strokes the same is like treating a paper cut the same as an amputated limb.
  * A **lacunar stroke** ($<5	ext{ cm}^3$) is a blockage of a tiny microscopic deep penetrating artery in the brain. It does not cause massive swelling and does not need invasive surgery; it needs blood pressure medication and aspirin (antiplatelet therapy).
  * A **large territorial stroke** ($>50	ext{ cm}^3$) is a complete blockage of a major trunk artery feeding half the brain. It causes massive swelling, pushes the brain across the skull (herniation), and requires immediate emergency endovascular catheter surgery. If Naive RAG mixes up these two treatments, the patient will die.
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
> $$D_{	ext{cosine}}(u, v) = 1 - rac{u \cdot v}{\|u\|_2 \|v\|_2}$$
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
* **👉 Visual Pointer Cues:** Point to the RRF formula on the left ($RRF(d) = \sum rac{1}{k + r(d)}$ with $k=60$). Then point to the Transformer joint attention diagram on the right (`[CLS] Query [SEP] Chunk`).

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 20 explains why hybrid search and cross-encoders are biologically and computationally necessary in medicine:
>
> **[Point to Left Card: Stage 1 Hybrid RRF]**  
> * Medical queries have a dual nature: they contain exact numerical drug dosages ('0.9 mg/kg', 'ASPECTS 6') which require lexical precision, and complex conceptual symptoms ('facial droop', 'wake-up stroke') which require semantic understanding.
> * Sparse BM25 excels at finding exact keywords like '0.9 mg/kg', but fails on synonyms. Dense PubMedBERT understands medical concepts, but struggles with exact numerical thresholds.
> * We merge both rankings using **Reciprocal Rank Fusion (RRF)** with smoothing constant $k = 60$:
> $$RRF\_Score(d) = \sum_{m \in \{	ext{BM25}, 	ext{Dense}\}} rac{1}{60 + r_m(d)}$$
> This generates a diverse, high-recall candidate pool of the top 15 candidate passages.
>
> **[Point to Right Card: Stage 2 Cross-Encoder]**  
> * In Stage 2, our GPU passes the candidate pool through `cross-encoder/ms-marco-MiniLM-L-6-v2`.
> * Unlike bi-encoders that encode query and document in isolation, the cross-encoder concatenates both into a single sequence: `[CLS] Query [SEP] Passage [EOS]`. Every layer of self-attention directly compares every query word against every passage word.
> * This achieves an extraordinary **82.5% clinical concept recall** across complex multi-guideline questions in only 51 milliseconds, completely eliminating irrelevant candidate chunks."



---

## PART 4: LANGGRAPH MULTI-AGENT ARCHITECTURE & DUAL-LLM ROUTING (SLIDES 21 – 28)

---

### SLIDE 21: LangGraph StateGraph & Cyclic Routing
* **Slide Title:** *LangGraph StateGraph & Cyclic Routing*
* **Slide Theme:** State machine graph visualization showing circular routing between Drafter and Critic nodes.
* **⏱️ Estimated Time:** 45 seconds
* **👉 Visual Pointer Cues:** Trace the state path: START $ightarrow$ Triage Agent $ightarrow$ Librarian Agent $ightarrow$ Drafter Agent $ightarrow$ Critic Agent. Emphasize the circular red loop that routes backwards from Critic to Drafter if validation fails.

#### 🎙️ Word-for-Word Spoken Script:
> "We now enter Part 4: our **LangGraph Multi-Agent Orchestration Architecture**.
>
> On Slide 21, we visualize our central control architecture: a cyclical state machine built on LangGraph's `StateGraph`.
>
> In production clinical AI, you cannot treat an LLM as a one-way conveyor belt. If an LLM generates a clinical report that accidentally drops a decimal point or recommends clot-busting medication to a patient on blood thinners, a linear pipeline will simply print the fatal error into the patient's medical record.
>
> In our architecture, the clinical draft is not the final word. It enters an adversarial closed-loop state machine where an independent Critic agent inspects the document against formal medical invariants. If any gate fails, the state machine halts certification, injects an explicit error trace into the shared state, and routes the report back to the Drafter for targeted surgical repair."

---

### SLIDE 22: State Machine Design & Why LangGraph
* **Slide Title:** *State Machine Design & Why LangGraph*
* **Cards Displayed:** Left Card (*LangGraph State Machine Design*) vs. Right Card (*Why Not Chains or One Prompt?*)
* **⏱️ Estimated Time:** 75 seconds
* **👉 Visual Pointer Cues:** Point to the typed state dictionary structure on the left card (`biomarkers`, `evidence_chunks`, `draft_report`, `error_trace`). Then contrast with the failure modes on the right card (Linear DAG collapse and Monolithic prompt drift).

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 22 details the formal state machine design and why standard agent frameworks fail:
>
> **[Point to Left Card: State Machine Design]**  
> * We define a strictly typed schema: `AgentState`. This state object carries immutable 3D biomarkers, parsed clinical history flags, retrieved guideline chunks, the current draft report, and an active error trace.
> * Every agent in our graph is implemented as an isolated, pure state-transition function:
> $$f_i: 	ext{AgentState} \longrightarrow \Delta 	ext{AgentState}$$
> * Nodes never call each other directly; they only mutate designated keys in `AgentState`.
> * The exit edge from the Critic Agent is a conditional branch: if `audit_passed` is True, it routes to `END` for physician signature. If False, it routes back to `Drafter` with the exact violation trace.
>
> **[Point to Right Card: Why Not Chains or Mega-Prompts?]**  
> * **Why not linear DAGs (like LangChain chains)?** Directed Acyclic Graphs cannot loop. If a linear chain makes a mistake on dosage or onset time, it has no computational mechanism to step backward and correct itself.
> * **Why not a single monolithic prompt?** Trying to tell a single LLM to segment an image, parse history, remember 13 guidelines, check dosages, and write a report causes catastrophic prompt dilution and instruction forgetting—resulting in a **28.4% clinical error rate**.
> * **Guaranteed Termination:** To prevent infinite loops, we enforce a strict bound: $	ext{max\_reflections} \le 2$. If a draft fails twice, a deterministic Python AST fallback rule enforces safety overrides, guaranteeing **100% finite convergence**."

#### 🔬 Plain-English Concept Explanations:
* **Biological / Medical Aspect (Attending Physician & Resident Doctor Analogy):**
  * In a hospital, a medical resident (trainee doctor) never sends a radiology report directly to the patient's chart. The resident writes a preliminary draft. Then, the senior Attending Neuroradiologist reviews it.
  * If the resident accidentally missed a subtle bleed or wrote the wrong stroke side, the Attending rejects the draft, marks the error in red pen, and tells the resident: *"Fix paragraph 3; the lesion is in the right hemisphere, not the left."* The resident fixes that exact sentence and brings it back. LangGraph replicates this exact hospital safety hierarchy in software.
* **Technical / Computational Aspect (What is a Stateful State Machine vs. Linear DAG?):**
  * **Linear DAG (Directed Acyclic Graph):** Like a factory conveyor belt moving in only one direction. If a product has a defect at Step 4, it falls off the end of the belt broken.
  * **State Machine (LangGraph):** Like a smart assembly station with a quality inspector. The inspector checks the product; if defective, a robotic arm swings it back to Step 3 with a sticky note explaining the defect, repairs it, and inspects it again before boxing it.

---

### SLIDE 23: The 4 Specialized NeuroAgents
* **Slide Title:** *The 4 Specialized NeuroAgents*
* **Slide Theme:** Visual card deck showcasing the 4 distinct agent personas: Triage, Librarian, Drafter, and Critic.
* **⏱️ Estimated Time:** 45 seconds
* **👉 Visual Pointer Cues:** Point to each agent card in sequence: Agent 1 (Triage / Vision), Agent 2 (Librarian / Knowledge), Agent 3 (Drafter / Synthesis), and Agent 4 (Critic / Safety Audit).

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 23 introduces the four specialized digital clinicians that constitute NeuroAgent:
>
> 1. **Agent 1: The Triage Agent:** Combines our 3D ACS-UNet perception model with our regex clinical history parser, acting as the emergency room frontline diagnostician.
> 2. **Agent 2: The Librarian Agent:** Oversees semantic and lexical knowledge retrieval across our ChromaDB guideline vector store, acting as the institutional medical reference librarian.
> 3. **Agent 3: The Drafter Agent:** A local, on-premise Large Language Model that synthesizes multimodal inputs into standardized American College of Radiology clinical reports.
> 4. **Agent 4: The Critic Agent:** A rigorous, adversarial auditor combining an independent LLM with deterministic Python Abstract Syntax Tree rules to certify safety."

---

### SLIDE 24: Node-Level Specification
* **Slide Title:** *Node-Level Specification*
* **Cards Displayed:** Left Card (*Nodes 1 & 2: Triage & Librarian*) vs. Right Card (*Nodes 3 & 4: Drafter & Critic*)
* **⏱️ Estimated Time:** 75 seconds
* **👉 Visual Pointer Cues:** Point to the exact data mutations on the left: how Triage populates `state['biomarkers']` and Librarian populates `state['evidence_chunks']`. Then point to the right showing how Drafter mutates `state['draft_report']` and Critic populates `state['error_trace']`.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 24 provides the exact software specification for each node:
>
> **[Point to Left Card: Nodes 1 & 2]**  
> * **Node 1 (Triage Agent):** Executes 5-fold ACS-UNet inference on the patient's 3D NIfTI scan and runs `ClinicalHistoryParser` on the doctor's clinical notes. It computes the deterministic boolean flags `ivt_eligible` and `evt_eligible`, writing the complete biomarker schema into `state['biomarkers']`.
> * **Node 2 (Librarian Agent):** Takes the computed lesion tier and onset time, constructs the compiled clinical query, executes hybrid BM25 + PubMedBERT RRF retrieval, runs the cross-encoder reranker, and stores the top-3 validated guideline excerpts in `state['evidence_chunks']`.
>
> **[Point to Right Card: Nodes 3 & 4]**  
> * **Node 3 (Drafter Agent):** Runs our local Llama-3-8B model at greedy temperature $T = 0.0$. It synthesizes the standard American College of Radiology report sections—Technique, Findings, Impression, and Recommendations—embedding bracketed numbered citations `[1]` through `[6]`.
> * **Node 4 (Critic Agent):** Executes our independent GPT-4o-mini auditor and runs 5 deterministic AST regex gates. If any gate fails, it sets `audit_passed = False`, appends the exact violation string to `state['error_trace']`, and routes back to Node 3. If all pass, it stamps `audit_passed = True` and terminates."

---

### SLIDE 25: Heterogeneous Dual-LLM Routing
* **Slide Title:** *Heterogeneous Dual-LLM Routing Architecture*
* **Slide Theme:** Architectural schematic contrasting the local edge deployment of Llama-3-8B with the cloud-based verification of GPT-4o-mini.
* **⏱️ Estimated Time:** 45 seconds
* **👉 Visual Pointer Cues:** Point to the left blue box labeled *Local Edge Device (Hospital Workstation)* containing ACS-UNet and Llama-3-8B. Contrast it with the right purple box labeled *Secure Cloud Verification Layer* containing GPT-4o-mini.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 25 illustrates our **Heterogeneous Dual-LLM Routing Architecture**.
>
> In production healthcare AI, relying on a single Large Language Model creates a critical systemic failure mode known as **self-preference bias**. If a language model drafts a clinical report containing an error, and you ask that same model: *'Is this report medically safe?'*, it will almost always agree with itself because it shares the same internal weight biases and blind spots.
>
> To eliminate this, NeuroAgent splits drafting and auditing across two completely different model families running in two isolated environments: an on-premise local Llama-3-8B Drafter, and an independent cloud GPT-4o-mini Critic. Let us examine the clinical and privacy rationale."

---

### SLIDE 26: Llama-3-8B Drafter + GPT-4o-mini Critic
* **Slide Title:** *Llama-3-8B Drafter + GPT-4o-mini Critic*
* **Cards Displayed:** Left Card (*Why Local Llama-3-8B [Ollama]?*) vs. Right Card (*Why GPT-4o-mini + AST as Critic?*)
* **⏱️ Estimated Time:** 90 seconds
* **👉 Visual Pointer Cues:** Point to the VRAM and privacy bullets on the left (4-bit GGUF, 4.7 GB VRAM, Zero PHI leakage). Then point to the adversarial audit and latency bullets on the right (No self-preference bias, <450 ms audit, 0.0% final drift).

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 26 explains the technical, clinical, and regulatory justification for our dual-LLM split:
>
> **[Point to Left Card: Local Llama-3-8B]**  
> 1. **Hospital Edge Efficiency:** By running Llama-3-8B-Instruct via Ollama in 4-bit quantized GGUF format, the model consumes only **4.7 Gigabytes of VRAM**. Paired with ACS-UNet (4.2 GB), both vision and language models fit comfortably on an 8 GB or 16 GB hospital workstation GPU.
> 2. **Absolute Patient Privacy (Zero PHI Egress):** Medical data privacy laws—including HIPAA in the US and the Digital Personal Data Protection (DPDP) Act in India—strictly forbid sending raw patient scans, identifiable hospital notes, or unredacted medical histories to external commercial APIs. By drafting 100% locally on hospital premises, zero patient health information ever leaves the physical building.
> 3. **Concise ACR Synthesis:** Llama-3-8B is constrained via system prompts to state exact numbers in the Findings section and summarize clinically in the Impression, eliminating repetitive 'broken-record' syndrome.
>
> **[Point to Right Card: GPT-4o-mini + AST Critic]**  
> 1. **Adversarial Cross-Family Audit:** Meta's Llama-3 architecture is audited by OpenAI's GPT-4 architecture paired with deterministic Python code. The Critic shares zero weights, zero attention patterns, and zero architectural blind spots with the Drafter.
> 2. **Speed and Cost:** Auditing a structured report with GPT-4o-mini takes under **450 milliseconds** and costs fractions of a cent per patient.
> 3. **The Result:** This adversarial partnership drives our final clinical hallucination rate to exactly **0.0%**."

#### 🔬 Plain-English Concept Explanations:
* **Biological & Regulatory Aspect (Why Patient Privacy Laws Matter in AI):**
  * Under healthcare laws (like HIPAA and DPDP), patient medical records (brain scans, names, medical histories) are strictly confidential **Protected Health Information (PHI)**. If a hospital doctor copies and pastes a patient's brain scan report into ChatGPT, that data is transmitted across the public internet to third-party corporate servers, violating federal law.
  * By running our Drafter on a local workstation inside the hospital with no internet connection required, NeuroAgent is 100% legally compliant by design.
* **Technical / Computational Aspect (What is Self-Preference Bias & 4-bit Quantization?):**
  * **Self-Preference Bias:** A proven cognitive flaw in LLMs. If an AI writes a sentence with a factual mistake, and you prompt the same AI to check its work, it will give itself an $A+$ grade because its own internal probability weights generated that sentence in the first place! Using a completely different model family acts as an unbiased second opinion.
  * **4-bit Quantization (GGUF):** In standard deep learning, every weight in an LLM is a 16-bit floating-point number (taking 16 GB of RAM for an 8-billion parameter model). Quantization compresses those numbers down to 4 bits with virtually zero loss in medical reasoning quality, shrinking the memory footprint from 16 GB down to 4.7 GB so it fits on a laptop GPU!

---

### SLIDE 27: The 5-Gate Neuro-Symbolic Critic & Reflection Loop
* **Slide Title:** *The 5-Gate Neuro-Symbolic Critic & Reflection Loop*
* **Metrics Displayed:** `94%` Pass on Iteration 1 | `100%` Converge within $\le 2$ iterations | `0.0%` Residual clinical hallucination
* **⏱️ Estimated Time:** 60 seconds
* **👉 Visual Pointer Cues:** Direct attention to the reflection loop diagram. Highlight the 94% first-pass bar, the 100% two-pass convergence, and the green 0.0% zero-drift badge.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 27 presents the performance of our **5-Gate Neuro-Symbolic Critic and Reflection Loop**.
>
> When the Drafter Agent outputs an initial draft report, it must pass through 5 strict verification gates before certification.
>
> Across our 50-patient clinical benchmark:
> * **94% of drafts pass all 5 gates on the very first iteration**, demonstrating the high precision of our temperature-zero prompting and modular RAG retrieval.
> * For the remaining 6% that encounter a minor discrepancy—such as a misplaced unit or a missing negative checklist item—our reflection loop feeds the error trace back to the Drafter.
> * **100% of all cases converge and pass within 2 or fewer iterations**, completely eliminating residual hallucinations."

---

### SLIDE 28: The 5 Verification Gates & Self-Correction
* **Slide Title:** *The 5 Verification Gates & Self-Correction Mechanics*
* **Cards Displayed:** Left Card (*Formal Specification of the 5 Gates*) vs. Right Card (*Iterative Self-Correction Mechanics*)
* **⏱️ Estimated Time:** 90 seconds
* **👉 Visual Pointer Cues:** Walk through the 5 gates on the left card in order: Gates 1 & 2 (Fidelity), Gate 3 (Safety Invariant), Gates 4 & 5 (Negative Checklist and References). Then trace the error trace correction loop on the right.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 28 details the formal mathematical specification of our 5 Verification Gates:
>
> **[Point to Left Card: Specification of the 5 Gates]**  
> 1. **Gate 1 (Volumetric Numerical Fidelity):** The Critic applies regular expressions to verify that the reported lesion volume matches our 3D tensor integration ($80.924	ext{ cm}^3$) and voxel count ($80,924$) exactly, with zero numerical drift.
> 2. **Gate 2 (Neuroanatomical Laterality & Symmetry):** Verifies that the reported lesion location matches the computed laterality ('Right Cerebral Hemisphere MCA') and that the asymmetry index ($\Delta_{	ext{sym}} = 0.0884$) is preserved.
> 3. **Gate 3 (Clinical Reperfusion Safety Invariant):** The absolute safety core. The Critic inspects the Recommendations section: if lesion volume exceeds $50.0	ext{ cm}^3$, or symptom onset exceeds $4.5	ext{ hours}$, or active anticoagulation is present, the report **must declare intravenous thrombolysis strictly contraindicated**. If a draft ever recommends tPA in violation of this, Gate 3 immediately fails.
> 4. **Gate 4 (Standard Intracranial Negative Checklist):** Enforces BIOSE hospital report standards, verifying that the report explicitly checks for the absence of acute intracranial hemorrhage, midline shift, and mass effect.
> 5. **Gate 5 (Academic Numbered Provenance):** Verifies that every clinical recommendation carries valid bracketed citations referencing our numbered guideline bibliography `[1]` through `[6]`.
>
> **[Point to Right Card: Iterative Self-Correction]**  
> If any gate fails, the Critic writes an explicit, machine-readable violation string into `state['error_trace']`—for example: *'GATE 3 VIOLATION: Infarct volume 80.924 cm³ exceeds 50.0 cm³ ceiling; intravenous thrombolysis must be labeled CONTRAINDICATED'*.
> On iteration $k+1$, this error trace is injected into the Drafter's prompt as an immutable repair constraint. The Drafter rewrites only the invalid clause, and the Critic re-evaluates, achieving 100% convergence in under 2 iterations."



---

## PART 5: CLINICAL OUTPUT, EMPIRICAL EVALUATION & ABLATIONS 4 & 5 (SLIDES 29 – 37)

---

### SLIDE 29: End-to-End Output — Certified Clinical Report
* **Slide Title:** *End-to-End Output: Certified Clinical Report*
* **Slide Theme:** Full-page visual mockup of the certified American College of Radiology (ACR) neuro-radiology report generated by NeuroAgent.
* **⏱️ Estimated Time:** 45 seconds
* **👉 Visual Pointer Cues:** Point to the structured sections in order: Header Banner with Facility and Scan ID, Temporal Triage Window, Technique, Findings (Volumetrics and Lateralization), Impression, and the numbered References [1]–[6] at the bottom.

#### 🎙️ Word-for-Word Spoken Script:
> "We now arrive at Part 5: our **Clinical Output, Comprehensive Benchmark Evaluation, and Final GPU Ablations**.
>
> On Slide 29, we display the ultimate deliverable of NeuroAgent: an end-to-end, certified hospital neuro-radiology report.
>
> Notice the uncompromising structural discipline of this document. It follows the exact American College of Radiology (ACR) reporting hierarchy:
> * A formal **Clinical Header** with temporal window categorization;
> * A **Technique** section documenting stereotaxic normalization and 3D volumetric analysis;
> * A **Findings** section detailing exact voxel volumetrics, laterality, contralateral asymmetry, and ASPECTS scoring;
> * An actionable **Impression and Recommendations** section providing definitive, guideline-backed reperfusion guidance; and
> * A numbered **Knowledge Base Bibliography** binding every claim to verified medical literature."

---

### SLIDE 30: Dynamic Triage & Numbered Provenance
* **Slide Title:** *Dynamic Triage & Numbered Provenance*
* **Cards Displayed:** Left Card (*Dynamic History & Biomarkers*) vs. Right Card (*Academic Provenance [1]–[6]*)
* **⏱️ Estimated Time:** 90 seconds
* **👉 Visual Pointer Cues:** Trace the temporal window changes on the left (2.0h $ightarrow$ 10.0h $ightarrow$ 50.0h $ightarrow$ Wake-Up). Point to the pyramidal decussation correlation. Then point to the clean citations on the right ([1] AHA/ASA 2026, [2] ESO 2021, [3] Radiopaedia ASPECTS).

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 30 highlights the dynamic responsiveness and provenance of our output:
>
> **[Point to Left Card: Dynamic Triage]**  
> NeuroAgent is not a static template; it dynamically recalculates its clinical reasoning whenever patient variables change:
> 1. **Temporal Re-Triage:** If the patient presents at **2.0 hours**, the system activates the hyperacute reperfusion protocol, recommending IV thrombolysis. If onset shifts to **10.0 hours**, it immediately contraindicates IV thrombolysis due to the 4.5-hour ceiling, but flags eligibility for mechanical thrombectomy under extended DAWN/DEFUSE-3 criteria. If onset is **50.0 hours**, it contraindicates both interventions, shifting to subacute edema management. If labeled a **Wake-Up Stroke**, it mandates the MRI DWI-FLAIR mismatch protocol.
> 2. **Clinico-Radiological Correlation (Pyramidal Decussation):** The human nervous system crosses in the medulla: motor tracts from the right cerebral cortex control the left side of the body. When our vision model detects a right-hemispheric stroke and the clinical history notes left-sided arm weakness, the system explicitly confirms: *'Concordant Pyramidal Decussation'*. If the history noted right-sided weakness, it alerts clinicians to an ipsilateral discrepancy!
>
> **[Point to Right Card: Numbered Provenance & Clean Prose]**  
> 1. **Zero Raw Code Leakage:** The report contains zero JSON artifacts, zero Python dictionaries, and zero file paths. It is 100% professional clinical English.
> 2. **Numbered Academic Citations:** Every single recommendation embeds clean bracketed numbers—such as `[1, 2]`—corresponding directly to peer-reviewed guidelines: `[1]` Prabhakaran AHA/ASA 2026, `[2]` Berge ESO 2021, and `[3]` Radiopaedia ASPECTS.
> 3. **Non-Redundant Prose:** Exact voxel counts ($80,924$) are documented once in Findings, while the Impression summarizes clinically, eliminating repetitive 'broken-record' phrasing."

#### 🔬 Plain-English Concept Explanations:
* **Biological / Medical Aspect (Pyramidal Decussation & Wake-Up Strokes):**
  * **Pyramidal Decussation:** At the base of the skull (the medulla oblongata), approximately 90% of all motor nerve fibers coming down from the cerebral cortex cross over to the opposite side of the spinal cord. Therefore, a stroke in the **right** hemisphere paralyzes the **left** side of the body. If an AI sees a right-brain stroke and the patient has right-arm paralysis, something is medically wrong (the patient either has a second hidden stroke or a non-vascular mimic). NeuroAgent catches this automatically.
  * **Wake-Up Stroke (DWI-FLAIR Mismatch):** When a patient goes to sleep normal at 10 PM and wakes up paralyzed at 6 AM, the exact onset time is unknown. Is the stroke 8 hours old or 30 minutes old? On MRI, **Diffusion-Weighted Imaging (DWI)** turns bright within minutes of ischemia because water gets trapped inside swelling cells. **FLAIR** MRI takes 4.5 hours to turn bright because water hasn't leaked into surrounding spaces yet. If DWI is bright but FLAIR is dark (a mismatch), the stroke is younger than 4.5 hours, and clot-busting medication can safely be administered!
* **Technical / Computational Aspect (Academic Citation Linking):**
  * In standard LLMs, if you ask for citations, they frequently invent fake paper titles, fabricate imaginary authors, and output dead URLs. In NeuroAgent, citations `[1]` through `[6]` are linked directly to actual PDF files sitting in ChromaDB that were mathematically retrieved and quoted in the report.

---

### SLIDE 31: Evaluation Datasets & Ground-Truth Protocol
* **Slide Title:** *Evaluation Datasets & Ground-Truth Protocol*
* **Cards Displayed:** Left Card (*ATLAS v3.0 + Healthy Control*) | Middle Card (*BIOSE Hospital MRI Corpus*) | Right Card (*RAGAS & Stroke QA Benchmark*)
* **⏱️ Estimated Time:** 75 seconds
* **👉 Visual Pointer Cues:** Point to the 3 dataset cards across the slide, highlighting the cohort sizes: 1,453 multi-center scans on the left, 34 hospital reports in the middle, and 50 clinical vignettes on the right.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 31 establishes the experimental rigor of our project: evaluating across three multi-center, real-world cohorts:
>
> 1. **The 3D Vision Cohort (ATLAS v3.0 + Controls):** The Anatomical Tracings of Lesions After Stroke benchmark comprising **1,453 high-resolution 3D T1-weighted MRIs** from multiple international imaging sites, with voxel-level lesion masks hand-traced by expert neuroradiologists ranging from 12 voxels to over 150 cubic centimeters. In addition, we evaluate on completely normal healthy brain controls to verify zero-false-positive specificity.
> 2. **The Clinical Report Corpus (BIOSE Dataset):** 40 brain MRI patient subjects with **34 real hospital radiology reports**, used to establish ground-truth clinical phrasing, negative intracranial checklists, and structural hierarchy.
> 3. **The Knowledge & RAGAS Benchmark:** A comprehensive suite of **50 complex acute stroke vignettes** covering hyperacute $\le 4.5	ext{h}$ presentations, 6–24h extended endovascular cases, wake-up strokes, lacunar small vessel disease, and active anticoagulation contraindications, evaluated against ground-truth consensus answers compiled from the 2026 AHA/ASA and ESO guidelines."

---

### SLIDE 32: Quantitative Results: RAGAS & Hallucination Ablation
* **Slide Title:** *Quantitative Results: RAGAS & Hallucination Ablation*
* **Metrics Displayed:** `0.96` Faithfulness | `0.94` Answer Relevance | `0.91` Context Precision | `0.93` Context Recall
* **⏱️ Estimated Time:** 60 seconds
* **👉 Visual Pointer Cues:** Point to the 4 prominent score dials at the top: Faithfulness (0.96), Answer Relevance (0.94), Context Precision (0.91), and Context Recall (0.93). Then point to the step-down hallucination bar chart below.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 32 presents our quantitative RAGAS evaluation and hallucination ablation results across our 50-patient clinical benchmark.
>
> NeuroAgent achieves world-class scores across all four standardized RAGAS dimensions:
> * **Faithfulness: 0.96**
> * **Answer Relevance: 0.94**
> * **Context Precision: 0.91**
> * **Context Recall: 0.93**
>
> More importantly, look at the bar chart below: we tracked clinical hallucination rates across four successive architectural configurations:
> * An unconstrained **Zero-Shot LLM** suffered a **28.4% clinical hallucination rate**.
> * Adding **Naive Linear RAG** reduced this to **14.6%**, but still suffered from context contamination.
> * Deploying our **4-Stage Modular RAG** dropped hallucinations to **4.2%**.
> * Finally, adding our **LangGraph 5-Gate Critic Reflection Loop eliminated all remaining errors**, achieving a verified **0.0% final clinical drift**. Let us examine what these metrics mean."

---

### SLIDE 33: RAGAS Performance & Hallucination Ablation Explained
* **Slide Title:** *RAGAS Performance & Hallucination Ablation Explained*
* **Cards Displayed:** Left Card (*RAGAS Results [n = 50]*) vs. Right Card (*Hallucination Ablation Breakdown*)
* **⏱️ Estimated Time:** 75 seconds
* **👉 Visual Pointer Cues:** Walk through the 4 RAGAS metrics on the left, explaining their exact clinical meaning. Then walk through the 4 stages of hallucination reduction on the right, ending on the closed-loop critic.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 33 breaks down what these scores mean in clinical terms:
>
> **[Point to Left Card: Clinical Meaning of RAGAS]**  
> * **Faithfulness (0.96):** Measures factual entailment. 96% of all clinical assertions made in the report are mathematically and semantically directly supported by the retrieved guideline text. The LLM is not making up facts.
> * **Answer Relevance (0.94):** Confirms that the report answers the exact clinical emergency at hand without rambling about irrelevant medical conditions.
> * **Context Precision (0.91):** Confirms that our hybrid search and cross-encoder place the most critical guideline passages right at the very top of the retrieved context window.
> * **Context Recall (0.93):** Confirms that 93% of all required guideline criteria (such as blood pressure limits, blood sugar limits, and dosing formulas) were successfully retrieved from ChromaDB.
>
> **[Point to Right Card: The Four Stages of Safety]**  
> * Moving from Zero-Shot (28.4%) to Naive RAG (14.6%) cut hallucinations in half by grounding the LLM in text.
> * Moving from Naive RAG to Modular RAG (4.2%) cut errors by another 70% by separating small vessel disease from large vessel occlusion.
> * Finally, our 5-Gate LangGraph Critic caught the remaining edge cases—such as minor unit mismatches—and forced targeted prompt rewrites, reaching **0.0% residual error**."

---

### SLIDE 34: 5 Empirical GPU Ablations: Dashboard
* **Slide Title:** *Complete 5-Panel Empirical GPU Ablation Dashboard*
* **Slide Theme:** Full 6-panel composite dashboard plot showing live hardware execution results across all 5 studies on our NVIDIA GeForce RTX 4060 Laptop GPU.
* **⏱️ Estimated Time:** 45 seconds
* **👉 Visual Pointer Cues:** Sweep across the 6 panels: Panel 1 (Chunking Integrity), Panel 2 (Embedding MRR), Panel 3 (Retrieval Pipeline), Panel 4 (Top-K Pareto Curve), Panel 5 (Temperature Drift), and Panel 6 (Critic Reflection Safety).

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 34 presents the master dashboard of our **5 Real Empirical GPU Ablation Studies**.
>
> Every single data point, bar chart, and curve shown on this dashboard was executed live on `cuda:0` using our **NVIDIA GeForce RTX 4060 Laptop GPU** across 232.6 seconds of total compute time.
>
> There is zero simulated, dummy, or mock data. We tested all 13 clinical guidelines and all 4,836 ChromaDB chunks on real hardware. Let us review the summary findings across each ablation."

---

### SLIDE 35: Findings from the 5 Ablations (RTX 4060)
* **Slide Title:** *Findings from the 5 Empirical Ablations (RTX 4060 GPU)*
* **Cards Displayed:** 1. Chunking (13 PDFs) | 2. Embeddings (4,836 chunks) | 3. Retrieval Pipeline | 4. Top-K Window | 5. Temperature & Critic
* **⏱️ Estimated Time:** 90 seconds
* **👉 Visual Pointer Cues:** Touch on each of the 5 numbered ablation summaries, highlighting the exact hardware numbers: 100% Recall@3 in Study 1, 7.73 ms in Study 2, 0.9167 MRR in Study 3, K=3 sweet spot in Study 4, and 0.0% drift in Study 5.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 35 synthesizes the core empirical findings from all 5 GPU studies:
>
> 1. **Ablation 1 (Chunking Strategy):** Title-aware chunking (1200 characters, 150 overlap) achieved **100.0% Recall@3** across 4,678 chunks, whereas Fixed-500 character splitting suffered from a catastrophic **0.8% syntactic boundary integrity**, severing 98% of medical clauses.
> 2. **Ablation 2 (Biomedical Embeddings on CUDA):** PubMedBERT (768 dimensions) achieved **100.0% Recall@3** and **0.8333 MRR** in only **7.73 milliseconds**, beating general MiniLM (0.7917 MRR) and BGE-small (0.7500 MRR).
> 3. **Ablation 3 (Retrieval Pipeline):** Performance scaled monotonically: Sparse BM25 (0.8125 MRR) $ightarrow$ Dense PubMedBERT (0.8333 MRR) $ightarrow$ Hybrid RRF (0.8542 MRR) $ightarrow$ and Hybrid RRF + Cross-Encoder (**0.9167 MRR, 87.5% Recall@1, 100.0% Recall@3**) in 51.69 ms.
> 4. **Ablation 4 (Top-K Context Window):** $K = 3$ is the Pareto sweet spot, delivering **100.0% precision and 88.9% F1** in 2,585 characters, whereas $K = 10$ bloated the prompt to 9,125 characters and diluted precision.
> 5. **Ablation 5 (Decoding Temperature & Critic):** Greedy decoding ($T = 0.0$) preserved **100.0% exact numerical volume and voxel counts**, whereas $T = 0.7$ mutated numbers in 60% of runs; our 5-Gate Critic reduced zero-shot safety violations from **80.0% down to 0.0%**."

---

### SLIDE 36: Context Window & GPU LLM Temperature (Visuals)
* **Slide Title:** *Context Window & GPU LLM Temperature Ablations*
* **Slide Theme:** Dual-plot visualization: Left plot showing Top-K Precision-Recall tradeoff curves; Right plot showing Temperature token mutation percentages.
* **⏱️ Estimated Time:** 45 seconds
* **👉 Visual Pointer Cues:** Point to the intersection peak at $K=3$ on the left plot. Then point to the green bar dropping and red error bar climbing as temperature increases from $T=0.0$ to $T=1.0$ on the right plot.

#### 🎙️ Word-for-Word Spoken Script:
> "On Slide 36, we look deeper into the experimental curves of **Ablations 4 and 5**.
>
> On the left, we plot Context Precision, Clinical Concept Recall, and Prompt Character Length across Top-K values of 1, 3, 5, and 10. Notice the clear Pareto inflection point at $K = 3$.
>
> On the right, we plot the effect of causal LLM decoding temperature on numerical token preservation and safety gate compliance using Qwen2.5 on our RTX 4060 GPU. Let us examine the theoretical mechanics behind these two curves."

---

### SLIDE 37: Top-K Optimization & Decoding Safety
* **Slide Title:** *Top-K Optimization & Decoding Safety*
* **Cards Displayed:** Left Card (*Ablation 4: Top-K Context Window Optimization*) vs. Right Card (*Ablation 5: GPU LLM Decoding & Critic Safety*)
* **⏱️ Estimated Time:** 90 seconds
* **👉 Visual Pointer Cues:** Point to the character counts on the left ($K=1$: 885 chars, $K=3$: 2,585 chars, $K=10$: 9,125 chars). Then point to the temperature comparison on the right ($T=0.0$ vs. $T=0.7$) and the 80% to 0% Critic drop.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 37 explains the critical engineering insights of our final two ablations:
>
> **[Point to Left Card: Top-K Context Window Optimization]**  
> * **The $K=1$ Deficit:** Retrieving only 1 chunk saves prompt space (885 characters), but concept recall collapses to **65.0%** (F1 score of 78.8%). A single chunk cannot contain both inclusion criteria AND contraindication lists.
> * **The $K=3$ Sweet Spot:** At $K = 3$ (2,585 characters), recall jumps to **80.0%**, Context Precision is a perfect **100.0%**, and the F1 score peaks at **88.9%**.
> * **The $K=5$ and $K=10$ Bloat:** Expanding to $K = 10$ inflates the prompt to **9,125 characters—a 3.5-fold overhead**—while Context Precision drops to **90.0%** because marginal, irrelevant passages dilute the context window.
>
> **[Point to Right Card: Temperature & Critic Safety]**  
> * **Greedy Decoding ($T = 0.0$):** In medical AI, temperature is not an artistic preference; it is a clinical safety dial. At $T = 0.0$, the model always picks the argmax token, preserving **100.0% exact floating-point volume and voxel strings** (`80.924 cm³`, `80,924 voxels`) across repeated evaluations.
> * **Stochastic Drift ($T \ge 0.3$):** As soon as temperature is raised to $T = 0.3$ or $T = 0.7$, the softmax distribution flattens, introducing random token sampling. In **20% to 60% of runs**, the model mutated numerical strings, rounding 80.924 into 80.9 or 81.
> * **The 5-Gate Critic Shield:** Unconstrained Zero-Shot generation suffered an **80.0% safety gate violation rate**. Naive RAG reduced this to 24.0%. Our **5-Gate Closed-Loop Critic drove violations to exactly 0.0%**, guaranteeing 100% gate compliance."



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
   - **Danger:** If administered to a patient with a massive stroke core ($>50	ext{ cm}^3$) or after 4.5 hours, the fragile dead blood vessels burst, causing fatal **hemorrhagic transformation** (bleeding into the brain).
5. **Mechanical Thrombectomy (Endovascular Treatment / EVT):**
   - A minimally invasive catheter procedure where an interventional neuroradiologist threads a tiny wire through an artery from the groin or wrist up into the brain, deploying a tiny wire cage (stent retriever) or suction tube to physically pull the clot out.
   - **Window:** Effective up to 6 hours routinely, and up to **24 hours** in selected patients with salvageable penumbra (proven by the DAWN and DEFUSE-3 trials).
6. **ASPECTS (Alberta Stroke Programme Early CT Score):**
   - A 10-point quantitative anatomical scoring system evaluating 10 key Middle Cerebral Artery (MCA) brain regions.
   - Normal brain = $10/10$. Each dead region subtracts 1 point. A score of $\le 5$ indicates extensive, severe irreversible damage.
7. **Pyramidal Decussation:**
   - The biological crossing of motor nerve fibers in the caudal medulla at the base of the brainstem. Corticospinal fibers from the left side of the brain cross over to control the right side of the body, and vice versa.
8. **Lacunar Infarcts & Small Vessel Disease (SVD):**
   - Small strokes ($<5	ext{ cm}^3$ or $<15	ext{ mm}$ in diameter) occurring in the deep, microscopic penetrating arteries of the brain. They cause focal symptoms (like isolated motor weakness) but do not cause massive brain swelling.
9. **Wake-Up Stroke & DWI-FLAIR Mismatch:**
   - When a patient wakes up with stroke symptoms and the exact onset time is unknown.
   - **Diffusion-Weighted Imaging (DWI)** turns bright within 15–30 minutes of cell death (cytotoxic edema).
   - **FLAIR MRI** takes 4.5 hours to turn bright (vasogenic edema).
   - If DWI is bright and FLAIR is normal (mismatch), the stroke occurred less than 4.5 hours ago, making the patient eligible for clot-busting medication!

---

### 💻 Computer Science, AI & Mathematical Terminology:
1. **3D Voxel:** A three-dimensional pixel. While a pixel has width and height ($\Delta x, \Delta y$), a voxel also has depth ($\Delta z$), representing a physical $1	ext{ mm}^3$ block of human brain tissue.
2. **Depthwise Separable 3D Convolution:**
   - A neural network optimization that splits a heavy $3 	imes 3 	imes 3$ convolution into a spatial-only depthwise convolution followed by a $1 	imes 1 	imes 1$ pointwise channel convolution, cutting mathematical operations by up to 90%.
3. **Dice Similarity Coefficient:**
   - A metric measuring the spatial overlap between two binary segmentations: $	ext{Dice} = rac{2 |A \cap B|}{|A| + |B|}$. A score of 1.0 indicates perfect overlap; 0.0 indicates zero overlap.
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
   * **Your Answer:** "Bilateral acute ischemic strokes represent less than 1.5% of all clinical stroke cases (usually basilar artery or embolic showers). Even in bilateral presentation, the lesions are almost never spatially symmetrical across the mid-sagittal plane, meaning the subtraction $|F(X) - 	ext{Flip}(F(X))|$ still generates strong non-zero asymmetric activation."

8. **Examiner Question:** *"Why is decoding temperature T = 0.0 mandatory in your pipeline?"*
   * **Your Answer:** "Our empirical Ablation 5 proved that any temperature above 0.0 introduces random probability sampling on floating-point tokens. At T = 0.7, numerical strings like 80.924 cm³ were mutated in 60% of runs. In medical AI, exact numerical fidelity is non-negotiable; greedy argmax decoding ($T=0.0$) guarantees 100% reproducible numerical fidelity."

9. **Examiner Question:** *"How did you handle the 4.5-hour time window in patients who wake up with stroke symptoms?"*
   * **Your Answer:** "Our Clinical History Parser explicitly detects wake-up and unwitnessed strokes. In accordance with the WAKE-UP trial protocol, standard time-based IV thrombolysis is flagged as contraindicated, and the system automatically prompts for an MRI DWI-FLAIR mismatch sequence to verify whether the tissue is under 4.5 hours old."

10. **Examiner Question:** *"What is the single most important metric proving your system is ready for clinical evaluation?"*
    * **Your Answer:** "Our 0.0% residual clinical drift across our 50-patient benchmark. While baseline Zero-Shot LLMs made safety violations in 28.4% of cases, our neuro-symbolic 5-gate critic caught and corrected every single violation, proving that AI can be bounded by deterministic clinical guardrails."
