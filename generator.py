"""
generator.py

This module handles the generation of grounded answers using the Google GenAI SDK.
It formats the retrieved chunks into a clear prompt context, sets rigid system rules
to prevent hallucination, and calls the Gemini 2.5 Flash model.
"""

import os
from google import genai  # We use the new, unified google-genai library as requested, NOT the legacy google-generativeai.
from google.genai import types  # Import types for defining system instructions and model configs.

def generate_answer(query: str, retrieved_chunks: list[dict], api_key: str = None) -> tuple[str, list[dict]]:
    """
    Generates a factually grounded answer using Gemini 2.5 Flash.
    
    Why:
    We enforce system instructions directing the LLM to restrict answers *only* to the
    provided context. This minimizes hallucination and makes the RAG system trustworthy.
    
    Args:
        query (str): The user's query question.
        retrieved_chunks (list[dict]): A list of chunk dicts containing text and source keys.
        api_key (str, optional): The Gemini API Key. If not supplied, reads from environment.
        
    Returns:
        tuple[str, list[dict]]: A tuple containing (generated_answer_text, list_of_used_chunks).
    """
    # 1. Fetch the API key. Fall back to the environment variable if none was provided in the arguments.
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        # Raise an informative error if the key isn't found anywhere, so the user knows to configure it
        raise ValueError("Gemini API key is missing. Please supply it or set it in the environment/dot-env.")
    
    # 2. Initialize the client using the new SDK syntax
    client = genai.Client(api_key=key)
    
    # 3. Construct the formatted context block by joining retrieved text chunks.
    # We prefix each chunk with its source filename and number to help the LLM refer to it.
    context_parts = []
    for idx, chunk in enumerate(retrieved_chunks, 1):
        source_name = chunk.get("source", "Unknown Document")
        chunk_text = chunk.get("text", "")
        # Create a clearly delineated boundary for each context block
        context_parts.append(f"--- Context Item [{idx}] (Source: {source_name}) ---\n{chunk_text}")
        
    # Join all context items into a single context string
    context_str = "\n\n".join(context_parts)
    
    # 4. Construct the prompt by embedding the context block and the query question
    prompt = (
        f"CONTEXT INFORMATION:\n"
        f"{context_str}\n\n"
        f"USER QUESTION:\n"
        f"{query}\n"
    )
    
    # 5. Define the strict grounding rules as system instructions.
    # System instructions tell the model how to behave, setting its persona and operational limits.
    system_instruction = (
        "Answer using ONLY the context below. "
        "If the answer is not in the context, say 'I don't have that "
        "information in the provided document.' Do not use any prior knowledge."
    )
    
    # 6. Configure the model settings.
    # We set temperature to 0.0 to make the output highly deterministic and focused on fact-extraction.
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.0
    )
    
    # 7. Call the Gemini API inside a try-except block to handle network, authentication, or safety errors gracefully.
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",  # Explicitly specify the requested gemini-2.5-flash model
            contents=prompt,
            config=config
        )
        # Return the generated text along with the retrieved chunks that served as the context
        return response.text, retrieved_chunks
        
    except Exception as e:
        # Wrap and raise the exception with clear error messages so the Streamlit UI can render it.
        raise RuntimeError(f"Gemini API Error: {str(e)}")
