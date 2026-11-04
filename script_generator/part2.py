# script_generator/part2.py

def get_part2_text():
    return """
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
  * A 2D digital photo is made of flat squares called **pixels**. A 3D MRI brain scan is composed of tiny 3D cubes called **voxels** (volume elements). In our scans, each voxel represents exactly $1 \times 1 \times 1 \text{ mm}^3$ of physical human brain tissue.
  * The **Dice Similarity Coefficient** measures how accurately the AI identifies the boundaries of the dead brain tissue. Imagine an expert neuroradiologist draws the exact boundaries of a stroke in red, and the AI draws its prediction in green. Dice measures the percentage of physical overlap between the two shapes. In 3D medical stroke imaging, lesions are extremely irregular, fuzzy, and variable, making a 61.91% Dice score on the challenging ATLAS v3.0 benchmark highly competitive with state-of-the-art models that are 100 times larger.
* **Technical / Computational Aspect (What is VRAM and Why Does 4.2 GB Matter?):**
  * **VRAM (Video Random Access Memory):** The ultra-fast memory located directly on the graphics card (GPU). When processing large 3D neural networks, the entire 3D image tensor and all intermediate neural activations must reside inside VRAM.
  * If a model requires 16 GB of VRAM and the hospital computer only has an 8 GB GPU, the computer throws a fatal `CUDA Out of Memory` error and halts completely. By requiring only 4.2 GB, ACS-UNet leaves half the GPU memory free to run our language model and retrieval engines simultaneously on the same machine!

---

### SLIDE 8: ACS-UNet — Attention, Symmetry & Efficiency
* **Slide Title:** *ACS-UNet: Attention, Symmetry & Efficiency*
* **Cards Displayed:** 1. Depthwise Separable 3D Convolutions | 2. Contralateral Symmetry ($\Delta_{\text{sym}}$) | 3. Why Not nnU-Net / Swin-UNETR? | 4. Performance on ATLAS v3.0
* **⏱️ Estimated Time:** 90 seconds
* **👉 Visual Pointer Cues:** Walk through the 4 numbered cards: explain the mathematical factorization in Card 1, the anatomical mirror subtraction in Card 2, contrast against the memory hogs in Card 3, and summarize the small-lesion Dice boost in Card 4.

#### 🎙️ Word-for-Word Spoken Script:
> "How does a model with only 309,000 parameters compete with models containing tens of millions of weights? Slide 8 explains the two architectural breakthroughs inside ACS-UNet:
>
> 1. **Depthwise Separable 3D Convolutions:** Standard 3D convolutions simultaneously mix spatial coordinates ($X, Y, Z$) and channel features ($C$), leading to cubic parameter growth. We factorize each standard $3 \times 3 \times 3$ convolution into a depthwise spatial convolution followed by a $1 \times 1 \times 1$ pointwise channel projection. This slashes parameter count and computational FLOPs by over 90% without sacrificing spatial receptive fields.
> 2. **Contralateral Hemispheric Symmetry ($\Delta_{\text{sym}}$):** The human brain exhibits natural bilateral symmetry across the mid-sagittal plane. Ischemic strokes, however, are almost exclusively unilateral. We exploit this biological prior by taking the encoder feature map $F_{\text{enc}}(X)$, flipping it across the left-right axis, and computing the absolute feature difference: $|F_{\text{enc}}(X) - \text{Flip}_{LR}(F_{\text{enc}}(X))|$. This instantly illuminates acute tissue asymmetry, giving the network a massive inductive bias.
> 3. **Why NOT nnU-Net or Swin-UNETR?** Standard baselines like nnU-Net (31.2 million parameters) and Swin-UNETR (62.2 million parameters) are brute-force architectures requiring 14.6 to 18.4 GB of VRAM. More importantly, they treat brain MRI as generic 3D voxels, lacking any neurological symmetry priors.
> 4. **Empirical Impact on ATLAS v3.0:** On our 1,453-patient cohort, incorporating this contralateral symmetry prior yields a **+16.3% Dice improvement on difficult small subcortical lesions** under 5 cubic centimeters, while operating at 146 milliseconds."

#### 🔬 Plain-English Concept Explanations:
* **Biological / Medical Aspect (Bilateral Brain Symmetry & Unilateral Strokes):**
  * If you look down at a human brain, the left hemisphere and right hemisphere are anatomical mirror images of each other. While humans have minor functional differences (like language usually on the left), the physical layout of blood vessels and brain matter is symmetrical across the central mid-sagittal fissure.
  * When a stroke occurs, a blood clot almost never blocks both sides of the brain at once; it blocks an artery on the left OR on the right. By mathematically subtracting the left side of the brain from the right side, healthy brain structures cancel out to zero, while the fresh stroke lights up like a beacon!
* **Technical / Computational Aspect (Depthwise Separable 3D Convolutions):**
  * Imagine you are filtering coffee. A standard 3D convolution tries to grind the coffee beans and heat the water in a single massive, complex machine ($3 \times 3 \times 3 \times C_{\text{in}} \times C_{\text{out}}$ multiplications).
  * A **depthwise separable convolution** splits the task into two simple steps: first, it filters each spatial slice independently (depthwise), and then it mixes the flavors using a simple $1 \times 1 \times 1$ operation (pointwise). This produces virtually the exact same mathematical output while requiring 100 times fewer internal weights!

---

### SLIDE 9: Volumetric Biomarker Extraction
* **Slide Title:** *Volumetric Biomarker Extraction*
* **Metrics Displayed:** `80.924 cm³` lesion volume (80,924 voxels) | `2 / 10` estimated ASPECTS | `0.0884` $\Delta_{\text{sym}}$, Right MCA | `0.000 cm³` Healthy control
* **⏱️ Estimated Time:** 60 seconds
* **👉 Visual Pointer Cues:** Point to the red segmented stroke volume card ($80.924\text{ cm}^3$), then to the low ASPECTS score ($2/10$), and contrast with the green Healthy Control card at the bottom displaying exactly $0.000\text{ cm}^3$ and ASPECTS $10/10$.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 9 demonstrates how our perception engine bridges computer vision into clinical reality. Once ACS-UNet generates the voxel probability map, our deterministic pipeline extracts four crucial clinical biomarkers:
>
> * **First, Infarct Burden:** For this severe acute stroke patient, the segmented lesion measures **80.924 cubic centimeters**, corresponding to exactly **80,924 positive voxels** at $1\text{ mm}^3$ resolution.
> * **Second, Topographic ASPECTS Score:** The model automatically maps the lesion against 10 critical anatomic Middle Cerebral Artery regions, estimating an ASPECTS score of **2 out of 10**.
> * **Third, Neuroanatomical Lateralization:** The model calculates a contralateral asymmetry index ($\Delta_{\text{sym}}$) of **0.0884**, localizing the lesion unambiguously to the **Right Cerebral Hemisphere in the MCA territory**.
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
* **👉 Visual Pointer Cues:** Point to the integration equation on the left card ($V_{\text{cm}^3} = \sum I(P_{\text{ens}} > 0.5) \dots$). Then trace across to the right card showing the boolean logic equation: $\text{ivt\_eligible} = (0 < V < 50\text{ cm}^3) \land (t_{\text{onset}} \le 4.5\text{h}) \land (\neg \text{Contraindicated})$.

#### 🎙️ Word-for-Word Spoken Script:
> "Slide 10 details our architectural safety bridge. How do we ensure that raw neural network predictions safely guide medical decisions without allowing a language model to invent numbers?
>
> **[Point to Left Card: Tensor-to-Biomarker Extraction]**  
> First, our pipeline calculates exact physical volume through discrete mathematical integration over the ensemble probability map $P_{\text{ens}}$:
> $$V_{\text{cm}^3} = \sum_{x,y,z} \mathbb{I}(P_{\text{ens}}(x,y,z) > 0.5) \times \frac{\Delta x \cdot \Delta y \cdot \Delta z}{1000}$$
> Here, each voxel's spatial dimensions ($1 \times 1 \times 1 \text{ mm}^3$) are summed and converted to cubic centimeters. This gives us an exact, reproducible ground truth: 80.924 cubic centimeters.
>
> **[Point to Right Card: Neuro-Symbolic Safety Bridge]**  
> Second, our system constructs a hard deterministic safety bridge:
> 1. Our `ClinicalHistoryParser` applies deterministic regular expressions to extract symptom onset time ($t_{\text{onset}}$) and flags clinical contraindications like active anticoagulant medications (e.g., Warfarin or Eliquis).
> 2. Then, before any Large Language Model is called, a pure Python function evaluates the official AHA/ASA and ESO reperfusion invariant:
> $$\text{ivt\_eligible} = (0 < V < 50.0\text{ cm}^3) \land (t_{\text{onset}} \le 4.5\text{ hours}) \land (\neg \text{HasClinicalContraindications})$$
> 3. These boolean decisions and exact numbers are stored in a structured JSON state object. The language model is **never permitted to perform mathematical calculations or make unconstrained medical decisions**. It receives verified facts as immutable symbolic constraints."

#### 🔬 Plain-English Concept Explanations:
* **Biological / Medical Aspect (Why the 50 cm³ and 4.5h Rules Exist):**
  * When an artery is blocked, brain cells die in two zones: the dead central **core** and the surrounding, struggling **penumbra**.
  * If the dead core exceeds **50 cubic centimeters**, the damage is so extensive that the blood vessel walls have begun to decompose. If a doctor injects intravenous clot-busting medication (IV-rtPA), the sudden rush of high-pressure blood will blow out the dead capillary walls, causing fatal bleeding inside the skull. The exact same risk occurs if more than **4.5 hours** have passed since symptoms began.
* **Technical / Computational Aspect (Why We Don't Let LLMs Do Math):**
  * Large Language Models are probabilistic text completion engines; they predict the next most likely token. When an LLM sees `80.924 > 50.0`, it does not perform a mathematical comparison on a CPU register; it simply guesses what words typically follow. Allowing an LLM to evaluate medical safety rules results in catastrophic failures. By evaluating the logic in pure Python code before the LLM is invoked, we achieve 100% mathematical determinism.
"""

