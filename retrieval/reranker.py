import torch
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class ClinFuseReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initializes the Cross-Encoder for precise re-ranking of retrieved passages/images.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CrossEncoder ({model_name}) on {self.device}...")
        self.model = CrossEncoder(model_name, device=self.device)
        print("CrossEncoder loaded successfully.")

    def rerank(self, query: str, fused_results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Re-ranks a list of fused results against the query.
        For images, it uses their associated radiology report text as a proxy.
        For text, it uses the chunk text directly.
        """
        if not fused_results:
            return []
            
        pairs = []
        for res in fused_results:
            modality = res.get("modality", "text")
            
            if modality == "image":
                # For images, we must score against their proxy text (the radiology report)
                # Assuming the report text is stored in metadata during indexing
                passage = res.get("metadata", {}).get("report_text", "")
            else:
                # For EHR text chunks
                passage = res.get("chunk_text", res.get("metadata", {}).get("chunk_text", ""))
                
            # If passage is empty (e.g. missing metadata), fallback to an empty string to avoid crashes
            pairs.append((query, passage))
            
        # Predict scores using the CrossEncoder
        scores = self.model.predict(pairs)
        
        # Attach scores to the results
        for i, res in enumerate(fused_results):
            # sentence-transformers predict returns raw logits or probabilities depending on model.
            # ms-marco-MiniLM outputs raw logits.
            res["rerank_score"] = float(scores[i])
            
        # Sort descending by CrossEncoder score
        fused_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return fused_results[:top_k]

if __name__ == "__main__":
    print("Testing Reranker Module...")
    reranker = ClinFuseReranker()
    
    # Mock fused results
    mock_results = [
        {"modality": "text", "chunk_text": "Patient has a history of smoking and lung cancer.", "id": "t1"},
        {"modality": "image", "metadata": {"report_text": "Normal chest x-ray, clear lungs."}, "id": "i1"},
        {"modality": "text", "chunk_text": "Patient complained of knee pain after a fall.", "id": "t2"}
    ]
    
    query = "Signs of pulmonary disease or smoking history."
    ranked = reranker.rerank(query, mock_results, top_k=3)
    
    print(f"\nQuery: {query}")
    for i, res in enumerate(ranked, 1):
        mod = res["modality"]
        text = res.get("chunk_text") if mod == "text" else res["metadata"]["report_text"]
        print(f"{i}. [{mod.upper()}] Score: {res['rerank_score']:.4f} | {text}")
