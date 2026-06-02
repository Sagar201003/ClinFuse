from typing import List, Dict, Any
from retrieval.chroma_store import ClinFuseVectorStore

class TemporalRetriever:
    def __init__(self, vector_store: ClinFuseVectorStore, temporal_index: dict = None):
        """
        Initializes the TemporalRetriever for longitudinal patient-specific RAG.
        """
        self.vector_store = vector_store
        # temporal_index is a dict mapping patient_id to a chronologically sorted list of their records
        self.temporal_index = temporal_index or {}

    def retrieve_with_history(self, query_embedding: Any, patient_id: str, 
                              current_date: str, top_k_current: int = 3, 
                              top_k_history: int = 2) -> List[Dict[str, Any]]:
        """
        Retrieves both current records and past historical records for a specific patient.
        """
        # Step 1: Get top_k_current results from current context
        # Querying both image and text collections filtered by patient_id
        img_results = self.vector_store.query_images(query_embedding, top_k=top_k_current, patient_id=patient_id)
        txt_results = self.vector_store.query_texts(query_embedding, top_k=top_k_current, patient_id=patient_id)
        
        current_results = []
        for r in img_results:
            r['temporal_label'] = "current"
            r['modality'] = "image"
            current_results.append(r)
            
        for r in txt_results:
            r['temporal_label'] = "current"
            r['modality'] = "text"
            current_results.append(r)
            
        # ChromaDB distance metrics mean lower score is better (cosine distance)
        current_results.sort(key=lambda x: x.get('score', float('inf')))
        current_results = current_results[:top_k_current]

        # Step 2: Get top_k_history results from PAST records only
        # query_temporal strictly filters for study_date < current_date
        history_results = self.vector_store.query_temporal(
            query_embedding=query_embedding, 
            patient_id=patient_id, 
            before_date=current_date, 
            top_k=top_k_history
        )
        
        # Step 3: Tag each historical result with its actual date
        for hr in history_results:
            date = hr.get("study_date", hr.get("metadata", {}).get("study_date", "unknown_date"))
            hr['temporal_label'] = f"historical - {date}"
            
        # Combine both lists
        combined_results = current_results + history_results
        return combined_results

    def build_temporal_context_string(self, temporal_results: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved results into a structured context string for the LLM generator,
        explicitly separating CURRENT evidence from HISTORICAL evidence.
        """
        context_parts = []
        
        for res in temporal_results:
            label = res.get("temporal_label", "unknown")
            
            if label == "current":
                header = "[CURRENT EVIDENCE]"
            else:
                # Extract the date from "historical - YYYY-MM-DD"
                date_str = label.split("-", 1)[1].strip() if "-" in label else label
                header = f"[HISTORICAL EVIDENCE - {date_str}]"
                
            meta = res.get("metadata", {})
            modality = res.get("modality", res.get("type", "text"))
            
            path = meta.get("image_path", "N/A") if modality == "image" else "N/A"
            report = meta.get("report_text", meta.get("chunk_text", "N/A"))
            
            part = f"{header}\nImage: {path} | Report: {report}\n"
            context_parts.append(part)
            
        return "\n".join(context_parts)

if __name__ == "__main__":
    print("Testing TemporalRetriever...")
    # Mock test logic could go here
    print("TemporalRetriever structure loaded successfully.")
