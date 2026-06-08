## What I Learned Building This 

Here are the 5 core architectural learnings gained from constructing this RAG application from scratch—highly useful for technical discussions and engineering interviews:

1. **Recursive Chunking Trade-Offs**:
   Using simple fixed-character chunking (e.g. slicing every 500 characters) frequently cuts sentences or words in half, damaging semantic context. Recursive character chunking mimics human readability by splitting on a hierarchy of logical boundaries (`\n\n` for paragraphs, `\n` for sections, `. ` for sentences, and ` ` for words). This keeps related arguments together in a single chunk, maximizing retrieval quality.

2. **Vector Space Model Matching**:
   The query vector and the document vectors must be embedded using the *exact same* model. If they are embedded using different models, their dimensions, feature weighting, and orientation will not align. Even a 384-dimensional vector from one model cannot be queried against a 384-dimensional vector of another, as they map completely different mathematical coordinate systems.

3. **Cosine Similarity vs. Euclidean Distance**:
   We used `IndexFlatIP` (Inner Product) rather than `IndexFlatL2` (Euclidean distance). L2 measures absolute straight-line distance, which is highly sensitive to text length (magnitude). By L2-normalizing our vectors to unit length prior to database insertion, the Inner Product calculation becomes mathematically identical to Cosine Similarity. Cosine similarity evaluates the angle between vectors, capturing semantic orientation and topic relevance independently of document length.

4. **Context Grounding Prompts (LLM Guardrails)**:
   Generic LLM inference draws on training sets, causing hallucinations when asked about specific company documents. To enforce grounding, we pass the retrieved text segments to the LLM and set a rigid System Instruction: `"Answer using ONLY the context below. If the answer is not in the context, say 'I don't have that information in the provided document.' Do not use any prior knowledge."` This overrides the model's pre-trained assumptions, keeping answers 100% verified.

5. **Retrieval Confidence Scoring**:
   Cosine similarity scores represent how closely the user's query aligns with the retrieved chunks. By taking the average similarity score of the top-4 chunks, we can display a "Retrieval Confidence" metric to the user. When the average similarity falls below 50%, it acts as a warning sign indicating that the uploaded documents likely do not contain the answer, signaling that the LLM response might be an empty default or low-assurance text.
