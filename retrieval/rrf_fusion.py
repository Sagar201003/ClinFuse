from typing import List, Dict, Any

def rrf_score(rank: int, k: int = 60) -> float:
    """
    Computes the Reciprocal Rank Fusion score for a given rank.
    (1-indexed rank)
    """
    return 1.0 / (k + rank)

def fuse(
    image_results: List[Dict[str, Any]], 
    dense_text_results: List[Dict[str, Any]], 
    bm25_results: List[Dict[str, Any]], 
    image_weight: float = 1.0, 
    text_weight: float = 1.0, 
    bm25_weight: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Fuses dense image, dense text, and sparse BM25 results into a single ranked list.
    Handles merging scores for items that appear in multiple lists (e.g., text chunks).
    """
    fused_scores = {}
    item_data = {}
    
    def process_list(results_list, weight, id_keys, default_modality):
        for rank, item in enumerate(results_list, start=1):
            # Attempt to extract an ID from the various possible keys
            item_id = None
            for key in id_keys:
                if key in item:
                    item_id = str(item[key])
                    break
            if not item_id:
                # fallback to string repr if no ID found
                item_id = str(item)
                
            # Determine modality
            modality = item.get("modality")
            if not modality:
                modality = item.get("metadata", {}).get("modality", default_modality)
                
            # Composite key ensures image and text with same ID don't collide
            unique_key = (item_id, modality)
            
            score = rrf_score(rank) * weight
            
            if unique_key not in fused_scores:
                fused_scores[unique_key] = 0.0
                
                # Store item payload, ensuring modality tag is explicitly retained at top level
                cloned_item = item.copy()
                cloned_item["modality"] = modality
                item_data[unique_key] = cloned_item
                
            fused_scores[unique_key] += score

    # Process each list, providing the expected ID keys
    process_list(image_results, image_weight, ["study_id", "id"], "image")
    process_list(dense_text_results, text_weight, ["chunk_id", "id"], "text")
    process_list(bm25_results, bm25_weight, ["chunk_index", "chunk_id", "id"], "text")
    
    # Reconstruct final fused list
    final_list = []
    for key, score in fused_scores.items():
        data = item_data[key]
        data["final_score"] = score
        final_list.append(data)
        
    # Sort descending by final fused score
    final_list.sort(key=lambda x: x["final_score"], reverse=True)
    return final_list
