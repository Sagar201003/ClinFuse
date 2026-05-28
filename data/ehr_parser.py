from typing import List, Dict, Any

def chunk_ehr_text(
    text: str, 
    patient_id: str, 
    study_id: str, 
    study_date: str, 
    max_tokens: int = 256, 
    overlap: int = 32
) -> List[Dict[str, Any]]:
    """
    Chunks EHR text into segments of max `max_tokens` with `overlap` tokens overlap.
    Returns a list of dicts containing the text chunk and its associated metadata.
    """
    # For data engineering purposes, we approximate tokens by whitespace-split words.
    # In production with an actual LLM, you would replace this with a proper tokenizer 
    # (e.g., from HuggingFace transformers or tiktoken).
    tokens = text.split()
    chunks = []
    chunk_index = 0
    
    if not tokens:
        return chunks

    step = max_tokens - overlap
    if step <= 0:
        raise ValueError("Overlap must be less than max_tokens.")

    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i : i + max_tokens]
        chunk_text = " ".join(chunk_tokens)
        
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "patient_id": str(patient_id),
                "study_id": str(study_id),
                "chunk_index": chunk_index,
                "study_date": study_date,
                "modality": "text"
            }
        })
        chunk_index += 1
        
        # Break if we've reached the end of the tokens
        if i + max_tokens >= len(tokens):
            break
            
    return chunks
