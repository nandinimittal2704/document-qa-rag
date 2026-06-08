"""
retriever.py

This module handles semantic retrieval by translating search queries into the
same embedding space as the indexed documents and executing similarity searches.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def retrieve(query: str, index: faiss.IndexFlatIP, chunks: list[dict], model: SentenceTransformer, k: int = 4) -> list[tuple[dict, float]]:
    """
    Translates the query into an embedding, normalizes it, and queries the FAISS
    index to find the top k most semantically similar chunks.
    
    Why:
    Using the exact same SentenceTransformer model for both the query and the documents
    is critical because they must exist in the identical high-dimensional vector space.
    If different models were used, their dimensions or coordinate systems would be 
    incompatible, making distance or similarity calculations completely meaningless.
    
    Args:
        query (str): The search query typed by the user.
        index (faiss.IndexFlatIP): The loaded FAISS vector index.
        chunks (list[dict]): The list of chunk dicts (keys: 'text', 'source').
        model (SentenceTransformer): The shared embedding model.
        k (int): Number of top documents to retrieve (default is 4).
        
    Returns:
        list[tuple[dict, float]]: A list of tuples containing (chunk_dictionary, similarity_score).
    """
    # 1. Encode the text query into a dense embedding vector using the shared model.
    # The input is wrapped as a list because SentenceTransformers expects a list of text inputs.
    query_vector = model.encode([query], show_progress_bar=False)
    
    # 2. Cast to float32 to match the format requirements of FAISS indices.
    query_vector = np.array(query_vector).astype('float32')
    
    # 3. L2-Normalize the query vector in place.
    # Why?
    # Because we normalize document embeddings prior to adding them to the FlatIP (Inner Product) index.
    # When both vectors are L2-normalized (length=1.0), the Inner Product search returns the dot product,
    # which is mathematically equivalent to Cosine Similarity.
    faiss.normalize_L2(query_vector)
    
    # 4. Search the FAISS index.
    # index.search takes the query vector and the number of top elements to retrieve (k).
    # It returns two 2D numpy arrays:
    # - scores: The inner product scores (cosine similarities) of the top k matches.
    # - indices: The integer database indices of those matches within the index.
    scores, indices = index.search(query_vector, k)
    
    results = []
    
    # 5. Extract and map the results back to our chunk list.
    # indices[0] contains the indices of the matched chunks.
    # scores[0] contains their similarity scores.
    for idx, score in zip(indices[0], scores[0]):
        # FAISS returns -1 for indices if it cannot find enough matches (e.g., in empty indices).
        # We also check that the index is within the bounds of our metadata list.
        if idx != -1 and idx < len(chunks):
            # Append a tuple containing the full chunk dictionary (text and source file) and the score.
            results.append((chunks[idx], float(score)))
    
    return results
