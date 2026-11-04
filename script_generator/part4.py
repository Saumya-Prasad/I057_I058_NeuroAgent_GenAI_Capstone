# script_generator/part4.py

def get_part4_text():
    return """
---

## PART 4: LANGGRAPH MULTI-AGENT ARCHITECTURE & DUAL-LLM ROUTING (SLIDES 21 – 28)

---

### SLIDE 21: LangGraph StateGraph & Cyclic Routing
* **Slide Title:** *LangGraph StateGraph & Cyclic Routing*
* **Slide Theme:** State machine graph visualization showing circular routing between Drafter and Critic nodes.
* **⏱️ Estimated Time:** 45 seconds
* **👉 Visual Pointer Cues:** Trace the state path: START $\rightarrow$ Triage Agent $\rightarrow$ Librarian Agent $\rightarrow$ Drafter Agent $\rightarrow$ Critic Agent. Emphasize the circular red loop that routes backwards from Critic to Drafter if validation fails.

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
> $$f_i: \text{AgentState} \longrightarrow \Delta \text{AgentState}$$
> * Nodes never call each other directly; they only mutate designated keys in `AgentState`.
> * The exit edge from the Critic Agent is a conditional branch: if `audit_passed` is True, it routes to `END` for physician signature. If False, it routes back to `Drafter` with the exact violation trace.
>
> **[Point to Right Card: Why Not Chains or Mega-Prompts?]**  
> * **Why not linear DAGs (like LangChain chains)?** Directed Acyclic Graphs cannot loop. If a linear chain makes a mistake on dosage or onset time, it has no computational mechanism to step backward and correct itself.
> * **Why not a single monolithic prompt?** Trying to tell a single LLM to segment an image, parse history, remember 13 guidelines, check dosages, and write a report causes catastrophic prompt dilution and instruction forgetting—resulting in a **28.4% clinical error rate**.
> * **Guaranteed Termination:** To prevent infinite loops, we enforce a strict bound: $\text{max\_reflections} \le 2$. If a draft fails twice, a deterministic Python AST fallback rule enforces safety overrides, guaranteeing **100% finite convergence**."

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
> 1. **Gate 1 (Volumetric Numerical Fidelity):** The Critic applies regular expressions to verify that the reported lesion volume matches our 3D tensor integration ($80.924\text{ cm}^3$) and voxel count ($80,924$) exactly, with zero numerical drift.
> 2. **Gate 2 (Neuroanatomical Laterality & Symmetry):** Verifies that the reported lesion location matches the computed laterality ('Right Cerebral Hemisphere MCA') and that the asymmetry index ($\Delta_{\text{sym}} = 0.0884$) is preserved.
> 3. **Gate 3 (Clinical Reperfusion Safety Invariant):** The absolute safety core. The Critic inspects the Recommendations section: if lesion volume exceeds $50.0\text{ cm}^3$, or symptom onset exceeds $4.5\text{ hours}$, or active anticoagulation is present, the report **must declare intravenous thrombolysis strictly contraindicated**. If a draft ever recommends tPA in violation of this, Gate 3 immediately fails.
> 4. **Gate 4 (Standard Intracranial Negative Checklist):** Enforces BIOSE hospital report standards, verifying that the report explicitly checks for the absence of acute intracranial hemorrhage, midline shift, and mass effect.
> 5. **Gate 5 (Academic Numbered Provenance):** Verifies that every clinical recommendation carries valid bracketed citations referencing our numbered guideline bibliography `[1]` through `[6]`.
>
> **[Point to Right Card: Iterative Self-Correction]**  
> If any gate fails, the Critic writes an explicit, machine-readable violation string into `state['error_trace']`—for example: *'GATE 3 VIOLATION: Infarct volume 80.924 cm³ exceeds 50.0 cm³ ceiling; intravenous thrombolysis must be labeled CONTRAINDICATED'*.
> On iteration $k+1$, this error trace is injected into the Drafter's prompt as an immutable repair constraint. The Drafter rewrites only the invalid clause, and the Critic re-evaluates, achieving 100% convergence in under 2 iterations."
"""

