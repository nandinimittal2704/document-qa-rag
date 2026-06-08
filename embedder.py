"""
embedder.py

This module handles the generation of text embeddings and building a vector 
search index using FAISS. Everything is implemented using raw Python and library
primitives, keeping the RAG pipeline transparent and easy to debug.
"""

import json
import os
import numpy as np
import faiss  # FAISS is selected for its high-performance C++ backend for vector similarity searches.
from sentence_transformers import SentenceTransformer  # SentenceTransformers is used to load open-source embedding models.

def build_and_save_index(chunks: list[dict], model: SentenceTransformer, index_path: str = "index.faiss", chunks_path: str = "chunks.json") -> None:
    """
    Creates embeddings for the text chunks, constructs a FAISS index, normalizes
    the embeddings, and saves both the index and chunk metadata to disk.
    
    Args:
        chunks (list[dict]): List of chunk dictionaries containing 'text' and 'source' filename.
        model (SentenceTransformer): The loaded embedding model.
        index_path (str): Path where the FAISS index will be saved.
        chunks_path (str): Path where the chunk metadata will be saved as JSON.
    """
    # 1. Extract the text strings from our list of dictionaries to pass to the embedding model
    texts = [chunk["text"] for chunk in chunks]
    
    # 2. Generate embeddings for all text chunks.
    # The output is a list of numpy arrays representing the high-dimensional vectors.
    embeddings = model.encode(texts, show_progress_bar=False)
    
    # 3. Cast the embedding array to float32.
    # FAISS requires input matrices to be float32 for optimal memory alignment and SIMD CPU instruction sets.
    embeddings_matrix = np.array(embeddings).astype('float32')
    
    # 4. L2-Normalize the embeddings in place.
    # Why normalize?
    # By normalizing all vectors to a unit length (L2 norm = 1.0), the dot product (Inner Product)
    # between any two vectors becomes mathematically equivalent to their Cosine Similarity.
    # Cosine similarity is defined as (A . B) / (||A|| * ||B||). If ||A|| = 1 and ||B|| = 1, then Cosine Similarity = A . B.
    faiss.normalize_L2(embeddings_matrix)
    
    # 5. Determine the vector dimensionality (for all-MiniLM-L6-v2, this is 384).
    dimension = embeddings_matrix.shape[1]
    
    # 6. Instantiate a FAISS IndexFlatIP (Flat Inner Product index).
    # Why IndexFlatIP instead of IndexFlatL2?
    # - IndexFlatL2 calculates Euclidean distance (straight-line distance).
    # - IndexFlatIP calculates Inner Product (dot product).
    #
    # When our embedding vectors are L2-normalized:
    # - Inner product (IP) = Cosine Similarity.
    # - Cosine similarity measures the angle/direction between vectors rather than their magnitude.
    # - This is highly desirable for text search because two documents of different lengths with 
    #   similar semantic meaning will align in direction, even if they have different raw term frequencies/lengths.
    # - Using IndexFlatIP ensures we query documents using Cosine Similarity, which is the standard
    #   and highest-performing similarity metric for dense text embeddings.
    index = faiss.IndexFlatIP(dimension)
    
    # 7. Add the L2-normalized embeddings matrix to the FAISS index.
    index.add(embeddings_matrix)
    
    # 8. Save the FAISS index file to disk.
    faiss.write_index(index, index_path)
    
    # 9. Save the chunk text and document name metadata to a JSON file.
    # This allows us to map the retrieved FAISS vector indices back to the actual text chunks and filenames.
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def load_index(index_path: str = "index.faiss", chunks_path: str = "chunks.json") -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    Loads both the FAISS vector index and corresponding chunk metadata from disk.
    
    Args:
        index_path (str): Filepath of the stored index.faiss.
        chunks_path (str): Filepath of the stored chunks.json.
        
    Returns:
        tuple: (faiss_index, list_of_chunk_dicts)
    """
    # Verify that both files exist before attempting to open them to prevent system-level file errors
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Missing required index files: {index_path} or {chunks_path}")
        
    # Read the serialized binary FAISS index from disk
    index = faiss.read_index(index_path)
    
    # Read the list of chunk dictionaries containing source metadata
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    return index, chunks
