import pickle
import re
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from pathlib import Path

def _tokenize(text: str) -> List[str]:
    """Simple tokenizer for BM25: lowercase and split by non-alphanumeric"""
    return re.findall(r'\w+', text.lower())

def build_index(chunks: List[str]) -> BM25Okapi:
    """
    Builds the BM25 index over all EHR text chunks.
    """
    tokenized_corpus = [_tokenize(chunk) for chunk in chunks]
    bm25_index = BM25Okapi(tokenized_corpus)
    return bm25_index

def search(query: str, chunks: List[str], bm25_index: BM25Okapi, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Tokenize query, get BM25 scores, and return top_k chunks 
    with their scores normalized to [0,1].
    """
    tokenized_query = _tokenize(query)
    scores = bm25_index.get_scores(tokenized_query)
    
    # Normalize scores to [0, 1]
    max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0
    normalized_scores = [float(score) / max_score for score in scores]
    
    # Get top k indices sorted by score
    top_indices = sorted(range(len(normalized_scores)), key=lambda i: normalized_scores[i], reverse=True)[:top_k]
    
    results = []
    for i in top_indices:
        if normalized_scores[i] > 0:
            results.append({
                "chunk_text": chunks[i],
                "score": normalized_scores[i],
                "chunk_index": i
            })
            
    return results

def persist_index(chunks: List[str], bm25_index: BM25Okapi, save_path: str = "bm25_index.pkl") -> None:
    """
    Persist chunk list alongside BM25 index using pickle.
    """
    with open(save_path, 'wb') as f:
        pickle.dump({
            "bm25_index": bm25_index,
            "chunks": chunks
        }, f)

def load_index(load_path: str = "bm25_index.pkl") -> Tuple[List[str], BM25Okapi]:
    """
    Load persisted chunk list alongside BM25 index using pickle.
    """
    path = Path(load_path)
    if not path.exists():
        raise FileNotFoundError(f"BM25 index file not found at {load_path}")
        
    with open(path, 'rb') as f:
        data = pickle.load(f)
        
    return data["chunks"], data["bm25_index"]
