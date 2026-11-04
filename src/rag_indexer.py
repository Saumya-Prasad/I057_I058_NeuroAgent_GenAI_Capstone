import os
import glob
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUIDELINES_DIR = os.path.join(BASE_DIR, "guildines")
if not os.path.exists(GUIDELINES_DIR):
    GUIDELINES_DIR = os.path.join(BASE_DIR, "guidelines")

VECTOR_DB_DIR = os.path.join(BASE_DIR, "data", "vector_store")
COLLECTION_NAME = "stroke_clinical_guidelines"
EMBEDDING_MODEL_NAME = "NeuML/pubmedbert-base-embeddings"

# ---------------------------------------------------------------------------
# Step 1: Real PDF Text Extractor
# ---------------------------------------------------------------------------
def extract_pages_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extracts text page by page from real clinical PDFs using PyMuPDF.
    Discards empty / unreadable pages.
    """
    filename = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)
    extracted = []
    
    for page_idx in range(len(doc)):
        text = doc[page_idx].get_text().strip()
        # Filter out short/empty covers or references headers
        if len(text) > 80:
            extracted.append({
                "source": filename,
                "page": page_idx + 1,
                "text": text
            })
    return extracted

# ---------------------------------------------------------------------------
# Step 2: Sliding-Window Chunker (Unit 3 Part 2, Slides 36-37)
# ---------------------------------------------------------------------------
def sliding_window_chunker(pages: list[dict], chunk_size_words: int = 250, overlap_words: int = 50) -> list[dict]:
    """
    Splits continuous document pages into overlapping text chunks.
    Preserves document name, page number, and section markers.
    """
    chunks = []
    chunk_counter = 0

    for item in pages:
        words = item["text"].split()
        stride = chunk_size_words - overlap_words
        
        for i in range(0, len(words), stride):
            chunk_words = words[i:i + chunk_size_words]
            if len(chunk_words) < 25:  # Ignore trailing tiny fragments
                continue
            
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "chunk_id": f"{item['source']}_p{item['page']}_c{chunk_counter}",
                "text": chunk_text,
                "metadata": {
                    "source": item["source"],
                    "page": int(item["page"]),
                    "word_count": len(chunk_words)
                }
            })
            chunk_counter += 1

    return chunks

# ---------------------------------------------------------------------------
# Step 3: Vector Indexing & ChromaDB Persistence (Unit 3 Part 1, Slides 23-36)
# ---------------------------------------------------------------------------
def build_vector_store():
    print("=" * 70)
    print("  NEUROAGENT: REAL CLINICAL GUIDELINES VECTOR INDEXER")
    print("=" * 70)

    pdf_files = sorted(glob.glob(os.path.join(GUIDELINES_DIR, "*.pdf")))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {GUIDELINES_DIR}")

    print(f"\n[1/4] Found {len(pdf_files)} authoritative guideline PDFs:")
    for p in pdf_files:
        print(f"   - {os.path.basename(p)}")

    # Extract all text
    print("\n[2/4] Extracting text across all pages...")
    all_pages = []
    for pdf_path in pdf_files:
        pages = extract_pages_from_pdf(pdf_path)
        all_pages.extend(pages)
        print(f"   Extracted {len(pages)} valid pages from {os.path.basename(pdf_path)}")

    # Chunking
    print("\n[3/4] Applying Sliding-Window Chunking (250 words, 50 words overlap)...")
    chunks = sliding_window_chunker(all_pages, chunk_size_words=250, overlap_words=50)
    print(f"   Total Chunks Generated: {len(chunks)}")

    # Load SentenceTransformer model
    print(f"\n[4/4] Encoding dense embeddings with '{EMBEDDING_MODEL_NAME}' & Indexing into ChromaDB...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

    # Delete if collection already exists to ensure fresh real build
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # Create collection with HNSW cosine similarity space
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 64
    total = len(chunks)
    for b_start in range(0, total, batch_size):
        b_end = min(b_start + batch_size, total)
        batch = chunks[b_start:b_end]
        
        b_texts = [c["text"] for c in batch]
        b_ids = [c["chunk_id"] for c in batch]
        b_metas = [c["metadata"] for c in batch]
        b_embeddings = embedder.encode(b_texts, batch_size=len(batch), show_progress_bar=False).tolist()

        collection.add(
            ids=b_ids,
            documents=b_texts,
            metadatas=b_metas,
            embeddings=b_embeddings
        )
        print(f"   Indexed {b_end}/{total} chunks ({(b_end/total)*100:.1f}%)")

    print("\n" + "=" * 70)
    print(f"  SUCCESS: {total} chunks indexed and persisted to '{VECTOR_DB_DIR}'!")
    print("=" * 70)

# ---------------------------------------------------------------------------
# Test Query Retrieval Sanity Check
# ---------------------------------------------------------------------------
def test_retrieval(query: str, top_k: int = 2):
    print(f"\n[TEST QUERY] \"{query}\"")
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    query_vec = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_vec, n_results=top_k)

    for i in range(top_k):
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i] if "distances" in results else None
        print(f"\n--- Result #{i+1} [Source: {meta['source']}, Page: {meta['page']}] (Cosine Distance: {dist:.4f}) ---")
        print(doc[:350] + "...")

if __name__ == "__main__":
    build_vector_store()
    
    # Run test queries matching your paper's tiers
    print("\n" + "#" * 70)
    print("  RUNNING CLINICAL SANITY CHECKS ON REAL RETRIEVAL")
    print("#" * 70)
    
    test_retrieval("STRIVE-2 small subcortical lacunar infarct definition diameter under 5 cm3", top_k=2)
    test_retrieval("intravenous alteplase thrombolysis time window 4.5 hours core infarct volume limit", top_k=2)
