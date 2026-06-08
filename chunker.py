"""
chunker.py

This module handles text extraction from PDFs and recursive character splitting
to prepare documents for vector embedding. It is written from scratch without
external RAG frameworks like LangChain or LlamaIndex to ensure full explainability.
"""

import re
import fitz  # PyMuPDF is selected because it is extremely fast and lightweight compared to PyPDF2 or pdfplumber.

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts plain text from a PDF document page-by-page using PyMuPDF (fitz).
    
    Why:
    We process page-by-page to ensure we can identify and handle empty pages or 
    scanned documents without OCR, while keeping memory overhead low.
    
    Args:
        file_path (str): The absolute or relative path to the PDF file.
        
    Returns:
        str: The full concatenated text of the PDF. Returns an empty string if
             the file is empty, corrupted, or has no extractable text.
    """
    # Initialize an empty list to accumulate text from each page.
    # Accumulating in a list and joining at the end is much more memory efficient in Python than string concatenation.
    text_parts = []
    
    try:
        # Open the PDF file using PyMuPDF
        doc = fitz.open(file_path)
        
        # Iterate through each page of the document sequentially
        for page_num in range(len(doc)):
            # Load the current page object
            page = doc.load_page(page_num)
            
            # Extract plain text using the default 'text' layout mode
            page_text = page.get_text()
            
            # Check if text was successfully extracted and is not just empty whitespace
            if page_text and page_text.strip():
                # Append the text, removing trailing whitespaces but preserving line structures
                text_parts.append(page_text.strip())
        
        # Close the document to release file handles and system resources immediately
        doc.close()
        
    except Exception as e:
        # If any file reading or parsing error occurs, catch it and return whatever was read so far,
        # or propagate/handle it gracefully in the calling context.
        print(f"Error reading PDF file: {e}")
        return ""
        
    # Join all pages with double newlines to maintain semantic boundaries between pages.
    return "\n\n".join(text_parts)


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[str]:
    """
    Splits text recursively based on a hierarchy of separators:
    Double Newlines -> Newlines -> Sentences -> Words -> Characters
    
    Why recursive character splitting:
    Unlike simple character chunkers, recursive splitting attempts to keep semantically related
    text together (e.g. paragraphs first, then sentences, then words) rather than cutting 
    words or sentences in half arbitrarily, leading to much better vector search retrieval.
    
    Args:
        text (str): The raw document text.
        chunk_size (int): The target maximum length of each chunk in characters.
        chunk_overlap (int): The number of characters that should overlap between consecutive chunks.
        
    Returns:
        list[str]: A list of chunk strings.
    """
    # The hierarchy of separators we use to split the text.
    # We order them from largest semantic unit (paragraphs) to smallest (characters)
    # 1. '\n\n' splits paragraphs.
    # 2. '\n' splits lines/headings.
    # 3. '. ' splits sentences (handled with regex to preserve punctuation).
    # 4. ' ' splits words.
    # 5. '' splits individual characters if everything else fails.
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    # We call our internal recursive splitting function
    return _split_recursive(text, separators, chunk_size, chunk_overlap)


def _split_recursive(text: str, separators: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Internal recursive helper function to split text.
    """
    # Strip leading and trailing whitespace from the text block.
    # Leading/trailing whitespace adds zero semantic value to vectors but takes up chunk space.
    text = text.strip()
    
    # Base case: if the text block is already within our chunk size limit, return it as a single chunk.
    if len(text) <= chunk_size:
        return [text]
    
    # Base case: if we have run out of separators, slice the text directly based on size and overlap.
    if not separators:
        chunks = []
        start = 0
        # Iterate through the text, incrementing our start index by (chunk_size - chunk_overlap)
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            # Move start index forward, leaving an overlap buffer
            start += (chunk_size - chunk_overlap)
            # Prevent infinite loops if chunk_size is smaller than overlap
            if chunk_size <= chunk_overlap or start >= len(text):
                break
        return chunks
    
    # Get the current separator to attempt splitting on, and keep the remaining separators for recursive calls
    current_sep = separators[0]
    remaining_seps = separators[1:]
    
    # Split the text by the current separator
    if current_sep == ". ":
        # Use a lookbehind regex pattern to split at periods, exclamation marks, or question marks
        # followed by whitespace. This keeps the punctuation attached to the preceding sentence.
        splits = re.split(r'(?<=[.!?])\s+', text)
    elif current_sep == "":
        # If empty string, split into individual characters
        splits = list(text)
    else:
        # Split literally by the separator string (e.g. '\n\n' or '\n' or ' ')
        splits = text.split(current_sep)
        
    # Clean up empty splits that result from consecutive separators (e.g., multiple spaces or newlines)
    splits = [s for s in splits if s]
    
    # Process each split component.
    # If any individual component is still larger than the target chunk_size,
    # we must recursively split it using the remaining separators.
    processed_splits = []
    for s in splits:
        if len(s) > chunk_size:
            # Recursively split the long component
            processed_splits.extend(_split_recursive(s, remaining_seps, chunk_size, chunk_overlap))
        else:
            processed_splits.append(s)
            
    # Reassemble the processed splits into chunks that are close to chunk_size but do not exceed it,
    # while respecting the chunk_overlap constraints.
    chunks = []
    current_pieces = []
    current_len = 0
    
    for piece in processed_splits:
        # Calculate the length we would add if we appended this piece to the current chunk.
        # If we already have pieces in the chunk, we must account for the separator character length.
        added_len = len(piece) + (len(current_sep) if current_pieces and current_sep != "" else 0)
        
        if current_len + added_len <= chunk_size:
            # The piece fits in the current chunk. Add it.
            current_pieces.append(piece)
            current_len += added_len
        else:
            # The piece does not fit. Finalize the current chunk first.
            if current_pieces:
                chunks.append(current_sep.join(current_pieces))
            
            # Backtrack to satisfy the chunk_overlap requirement.
            # We pop elements from the start of current_pieces until the remaining text is within the overlap limit.
            while current_pieces:
                # If we have multiple pieces, we check what the overlap length would be if we remove the oldest piece.
                if len(current_pieces) > 1:
                    overlap_len = sum(len(p) for p in current_pieces[1:]) + len(current_sep) * (len(current_pieces) - 2)
                    if overlap_len <= chunk_overlap:
                        current_pieces = current_pieces[1:]
                        current_len = overlap_len
                        break
                    else:
                        current_pieces = current_pieces[1:]
                else:
                    # If only 1 piece remains and it exceeded the overlap, we must clear the list and reset length.
                    current_pieces = []
                    current_len = 0
                    break
            
            # Add the new piece to the fresh (or overlapping) chunk
            current_pieces.append(piece)
            current_len += len(piece) + (len(current_sep) if current_pieces and current_sep != "" and current_len > 0 else 0)
            
    # Append the last remaining pieces if there are any left.
    if current_pieces:
        chunks.append(current_sep.join(current_pieces))
        
    return chunks
