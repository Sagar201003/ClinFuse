import re
import numpy as np
from typing import List, Dict, Any

def split_into_claims(response_text: str) -> List[str]:
    """
    Splits the LLM response by sentence boundaries and filters out
    short sentences (less than 10 words).
    """
    # Split by common sentence terminators followed by whitespace
    raw_sentences = re.split(r'(?<=[.!?])\s+', response_text)
    claims = []
    
    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue
        # Filter sentences shorter than 10 words
        if len(sent.split()) >= 10:
            claims.append(sent)
            
    return claims

def score_claim(claim: str, retrieved_chunks: List[Dict[str, Any]], text_encoder: Any) -> Dict[str, Any]:
    """
    Scores a single claim against all retrieved evidence chunks using cosine similarity.
    Assumes text_encoder outputs L2-normalized embeddings, so cosine similarity is just the dot product.
    """
    # 1. Encode the claim
    claim_emb = text_encoder.encode_text(claim)
    
    # 2. Extract text from retrieved chunks
    chunk_texts = []
    for chunk in retrieved_chunks:
        # Handle both image (proxy report) and text (EHR chunk) modalities
        modality = chunk.get("modality", chunk.get("type", "text"))
        meta = chunk.get("metadata", {})
        
        if modality == "image":
            text = meta.get("report_text", "")
        else:
            text = chunk.get("chunk_text", meta.get("chunk_text", ""))
            
        # Fallback if empty
        if not text.strip():
            text = "No valid text content available."
            
        chunk_texts.append(text)
        
    if not chunk_texts:
        # Edge case: no evidence provided
        return {
            "claim": claim,
            "max_similarity": 0.0,
            "best_evidence": "",
            "confidence_label": "LOW",
            "grounded": False
        }
        
    # 3. Encode all evidence texts in batch
    evidence_embs = text_encoder.encode_texts(chunk_texts)
    
    # 4. Compute similarities (dot product)
    similarities = [np.dot(claim_emb, ev_emb) for ev_emb in evidence_embs]
    
    max_idx = int(np.argmax(similarities))
    max_sim = float(similarities[max_idx])
    best_ev = chunk_texts[max_idx]
    
    # 5. Apply Thresholding
    if max_sim >= 0.75:
        confidence = "HIGH"
        grounded = True
    elif max_sim >= 0.5:
        confidence = "MEDIUM"
        grounded = False
    else:
        confidence = "LOW"
        grounded = False
        
    return {
        "claim": claim,
        "max_similarity": max_sim,
        "best_evidence": best_ev,
        "confidence_label": confidence,
        "grounded": grounded
    }

def score_response(response_text: str, retrieved_chunks: List[Dict[str, Any]], text_encoder: Any) -> Dict[str, Any]:
    """
    Parses an entire LLM response, scores all claims, and returns an aggregate hallucination risk.
    """
    claims = split_into_claims(response_text)
    
    if not claims:
        # If no valid claims found (e.g. response was extremely short), default return
        return {
            "claim_scores": [],
            "overall_confidence": 0.0,
            "hallucination_risk": "HIGH",
            "ungrounded_claims": []
        }
        
    claim_scores = []
    ungrounded_claims = []
    total_similarity = 0.0
    
    # Process each claim
    for claim in claims:
        score_dict = score_claim(claim, retrieved_chunks, text_encoder)
        claim_scores.append(score_dict)
        total_similarity += score_dict["max_similarity"]
        
        if not score_dict["grounded"]:
            ungrounded_claims.append(claim)
            
    # Calculate macro confidence
    mean_sim = total_similarity / len(claims)
    
    # Determine overall hallucination risk (inverse of confidence)
    if mean_sim >= 0.75:
        risk = "LOW"
    elif mean_sim >= 0.5:
        risk = "MEDIUM"
    else:
        risk = "HIGH"
        
    return {
        "claim_scores": claim_scores,
        "overall_confidence": float(mean_sim),
        "hallucination_risk": risk,
        "ungrounded_claims": ungrounded_claims
    }
