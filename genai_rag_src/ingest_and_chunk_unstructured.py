"""
================================================================================
  NEUROAGENT: CLINICAL PDF INGESTION & SEMANTIC CHUNKING WITH UNSTRUCTURED & CHROMADB
================================================================================
This script implements Step 1 of the Modular RAG Pipeline:
1. Ingests official stroke guidelines (AHA/ASA, STRIVE-2, ESO, TOAST) from 'guildines/'
2. Extracts document structure and parses into Unstructured Document Elements
3. Applies 'chunk_by_title' semantic chunking (groups paragraphs under section headers)
4. Encodes dense vectors using SentenceTransformers (all-MiniLM-L6-v2)
5. Persists the knowledge base into ChromaDB with HNSW Cosine Indexing
================================================================================
"""

import os
import re
import glob
import fitz  # PyMuPDF for high-fidelity text and layout block extraction
import chromadb
from sentence_transformers import SentenceTransformer

# Unstructured elements and title-based chunking
from unstructured.documents.elements import (
    Title,
    NarrativeText,
    ListItem,
    Table,
    ElementMetadata
)
from unstructured.chunking.title import chunk_by_title
from unstructured.cleaners.core import clean_extra_whitespace, clean_dashes

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUIDELINES_DIR = os.path.join(BASE_DIR, "guildines")
if not os.path.exists(GUIDELINES_DIR):
    GUIDELINES_DIR = os.path.join(BASE_DIR, "guidelines")

VECTOR_DB_DIR = os.path.join(BASE_DIR, "data", "vector_store")
COLLECTION_NAME = "stroke_clinical_guidelines"
EMBEDDING_MODEL_NAME = "NeuML/pubmedbert-base-embeddings"

# ---------------------------------------------------------------------------
# Step 1: Structural Extraction into Unstructured Elements
# ---------------------------------------------------------------------------
def extract_unstructured_elements_from_pdf(pdf_path: str) -> list:
    """
    Parses a PDF into typed Unstructured Elements (Title, NarrativeText, ListItem)
    by analyzing font sizes, paragraph headers, and line structures.
    """
    doc = fitz.open(pdf_path)
    filename = os.path.basename(pdf_path)
    elements = []

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        blocks = page.get_text("blocks")  # Extract layout blocks: (x0, y0, x1, y1, text, block_no, block_type)
        
        for b in blocks:
            raw_text = b[4].strip()
            if not raw_text or len(raw_text) < 15:
                continue

            # Clean whitespace, hyphenation, and web-printed PDF navigation headers (e.g., Radiopaedia menus)
            cleaned_text = clean_dashes(clean_extra_whitespace(raw_text))
            cleaned_text = re.sub(r'(?:[A-Z]\s+){2,}[A-Z]\s*\([^\)]*\)', '', cleaned_text)
            cleaned_text = re.sub(r'\(\/[a-zA-Z0-9_\-\/\?\=\&\%\+\s\.]*\)', '', cleaned_text)
            cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()
            if not cleaned_text or len(cleaned_text) < 25 or "L O G I N" in raw_text or "C O U R S E S" in raw_text and len(cleaned_text) < 80:
                continue
            
            # Detect whether this block represents a Section Header / Title
            lines = [l.strip() for l in cleaned_text.split("\n") if l.strip()]
            first_line = lines[0] if lines else ""
            is_title = False

            # Header heuristics: short length, uppercase, or starts with section numbering (e.g. "3.1", "Table 2", "Section")
            if len(first_line) < 100 and (
                first_line.isupper() or 
                any(first_line.startswith(prefix) for prefix in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "Table", "Section", "Part", "Recommendation", "STRIVE"])
            ):
                is_title = True

            meta = ElementMetadata(
                filename=filename,
                page_number=page_num
            )

            if is_title and len(cleaned_text) < 150:
                elements.append(Title(text=cleaned_text, metadata=meta))
            elif cleaned_text.startswith("- ") or cleaned_text.startswith("• "):
                elements.append(ListItem(text=cleaned_text, metadata=meta))
            else:
                elements.append(NarrativeText(text=cleaned_text, metadata=meta))

    return elements

# ---------------------------------------------------------------------------
# Step 2B: Ingest Real Hospital Diagnosis Reports from BIOSE Dataset
# ---------------------------------------------------------------------------
def extract_biose_clinical_reports() -> list[dict]:
    """
    Ingests 34 real hospital patient reports from the BIOSE Brain MRI dataset.
    Extracts structured fields (Clinical History, Technique, Findings, Impression).
    """
    biose_path = os.path.join(BASE_DIR, "data", "biose_reports", "BIOSE MRI A Multimodal Brain MRI Dataset with Clin", "2.Cleaned_Patient_Diagnosis_Report.xlsx")
    if not os.path.exists(biose_path):
        return []

    import pandas as pd
    df = pd.read_excel(biose_path)
    biose_chunks = []

    print(f"\n[1.5/3] Ingesting {len(df)} Real Clinical Reports from BIOSE Brain MRI Dataset...")
    for idx, row in df.iterrows():
        sub_id = str(row.get("participant_id", f"sub-{idx:02d}"))
        category = str(row.get("Diagnosis_Category", "General"))
        history = str(row.get("ClinicalHistory", "Unknown"))
        findings = str(row.get("Findings", ""))
        impression = str(row.get("Impression", ""))

        if not findings or findings == "nan":
            continue

        report_text = f"BIOSE HOSPITAL CLINICAL REPORT - {sub_id} ({category})\nCLINICAL INDICATION: {history}\nFINDINGS:\n{findings}\nIMPRESSION:\n{impression}"
        
        biose_chunks.append({
            "chunk_id": f"BIOSE_{sub_id}_report",
            "text": report_text,
            "metadata": {
                "source": f"BIOSE_Hospital_Report_{sub_id}.xlsx",
                "page": 1,
                "char_count": len(report_text),
                "category": category
            }
        })

    print(f"      Formed {len(biose_chunks)} structured hospital clinical report chunks from BIOSE.")
    return biose_chunks

# ---------------------------------------------------------------------------
# Step 2: Semantic Chunking via Unstructured 'chunk_by_title'
# ---------------------------------------------------------------------------
def process_guidelines_with_unstructured(max_chars: int = 1200, combine_under: int = 300) -> list[dict]:
    """
    Uses Unstructured's chunk_by_title algorithm:
    - Never breaks section titles away from their accompanying clinical text.
    - Combines small sub-paragraphs under the same section up to 'max_chars'.
    - Preserves source PDF and page number provenance for audit trails.
    """
    pdf_files = sorted(glob.glob(os.path.join(GUIDELINES_DIR, "*.pdf")))
    if not pdf_files:
        raise FileNotFoundError(f"No clinical guideline PDFs found in: {GUIDELINES_DIR}")

    all_processed_chunks = []
    chunk_global_id = 0

    print(f"\n[1/3] Processing {len(pdf_files)} guideline PDFs with Unstructured Semantic Chunking...")

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"   -> Extracting structural elements: {filename}")
        elements = extract_unstructured_elements_from_pdf(pdf_path)
        print(f"      Extracted {len(elements)} structural elements.")

        # Apply Unstructured chunk_by_title
        semantic_chunks = chunk_by_title(
            elements,
            max_characters=max_chars,
            combine_text_under_n_chars=combine_under,
            new_after_n_chars=int(max_chars * 0.8),
            multipage_sections=True
        )
        print(f"      Formed {len(semantic_chunks)} semantic chunks via chunk_by_title.")

        for sc in semantic_chunks:
            # Extract page number and metadata
            page_num = getattr(sc.metadata, "page_number", 1)
            chunk_text = sc.text.strip()
            
            if len(chunk_text) < 40:
                continue

            all_processed_chunks.append({
                "chunk_id": f"{filename}_chunk_{chunk_global_id:05d}",
                "text": chunk_text,
                "metadata": {
                    "source": filename,
                    "page": int(page_num) if page_num else 1,
                    "char_count": len(chunk_text)
                }
            })
            chunk_global_id += 1

    return all_processed_chunks

# ---------------------------------------------------------------------------
# Step 3: Embed & Index into ChromaDB
# ---------------------------------------------------------------------------
def index_into_chromadb(chunks: list[dict]):
    print(f"\n[2/3] Loading Sentence-Transformer '{EMBEDDING_MODEL_NAME}' on local GPU/CPU...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"\n[3/3] Initializing ChromaDB Persistent Storage at: '{VECTOR_DB_DIR}'...")
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

    # Recreate collection to ensure clean state
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # Unit 3 HNSW Cosine Metric
    )

    batch_size = 64
    total_chunks = len(chunks)
    print(f"   Encoding and indexing {total_chunks} chunks in batches of {batch_size}...")

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metas = [c["metadata"] for c in batch]
        
        embeddings = embedder.encode(texts, batch_size=len(batch), show_progress_bar=False).tolist()

        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metas,
            embeddings=embeddings
        )
        print(f"   Indexed [{min(i + batch_size, total_chunks)}/{total_chunks}] ({(min(i + batch_size, total_chunks) / total_chunks)*100:.1f}%)")

    print(f"\n SUCCESS: Indexed {total_chunks} Unstructured chunks into ChromaDB!")

# ---------------------------------------------------------------------------
# Verification / Query Testing
# ---------------------------------------------------------------------------
def verify_retrieval(query: str, top_k: int = 2):
    print("\n" + "=" * 70)
    print(f"  VERIFYING RETRIEVAL FOR: \"{query}\"")
    print("=" * 70)
    
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    query_vec = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_vec, n_results=top_k)

    for idx in range(top_k):
        doc = results["documents"][0][idx]
        meta = results["metadatas"][0][idx]
        dist = results["distances"][0][idx] if "distances" in results else 0.0
        print(f"\n[Rank #{idx + 1}] Source: {meta['source']} | Page: {meta['page']} | Cosine Dist: {dist:.4f}")
        print("-" * 70)
        # Safe ASCII / UTF-8 print for Windows consoles
        safe_text = doc[:350].encode('ascii', errors='replace').decode('ascii')
        print(safe_text + "...")

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    # 1. Chunk all 12 Guidelines with Unstructured
    guideline_chunks = process_guidelines_with_unstructured(max_chars=1200, combine_under=300)
    print(f"\nTotal Semantic Chunks from 12 PDFs: {len(guideline_chunks)}")

    # 2. Extract Real Hospital Reports from BIOSE
    biose_chunks = extract_biose_clinical_reports()

    all_chunks = guideline_chunks + biose_chunks
    print(f"\nTOTAL KNOWLEDGE BASE CHUNKS TO INDEX: {len(all_chunks)} (12 Guidelines + 34 BIOSE Reports)")
    
    # 3. Index into ChromaDB
    index_into_chromadb(all_chunks)

    # 4. Test Retrieval across disparate medical domains
    verify_retrieval("STRIVE-2 small subcortical infarct lacunar lesion volume under 5 cm3 criteria", top_k=2)
    verify_retrieval("DAWN DEFUSE 3 endovascular thrombectomy EVT extended time window ASPECTS", top_k=2)
    verify_retrieval("BIOSE hospital acute infarct right frontal lobe restricted diffusion DWI", top_k=2)
