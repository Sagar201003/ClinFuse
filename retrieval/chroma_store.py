import chromadb
from typing import List, Dict, Any, Optional

class ClinFuseVectorStore:
    def __init__(self, db_path: str = "./chroma_db"):
        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Create TWO separate collections
        # Using cosine similarity as it is standard for L2-normalized embeddings
        self.image_collection = self.client.get_or_create_collection(
            name="image_collection",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.text_collection = self.client.get_or_create_collection(
            name="text_collection",
            metadata={"hnsw:space": "cosine"}
        )

    def add_image(self, study_id: str, embedding: Any, metadata: Dict[str, Any]) -> None:
        """
        Stores image embeddings.
        Expected metadata: patient_id, study_id, image_path, study_date, modality="image"
        """
        metadata["modality"] = "image"
        
        # Ensure embedding is a list
        emb_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        
        self.image_collection.add(
            ids=[str(study_id)],
            embeddings=[emb_list],
            metadatas=[metadata]
        )

    def add_text_chunk(self, chunk_id: str, embedding: Any, metadata: Dict[str, Any]) -> None:
        """
        Stores EHR text chunk embeddings.
        Expected metadata: patient_id, study_id, chunk_index, study_date, modality="text", chunk_text
        """
        metadata["modality"] = "text"
        
        emb_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        
        self.text_collection.add(
            ids=[str(chunk_id)],
            embeddings=[emb_list],
            metadatas=[metadata]
        )

    def query_images(self, query_embedding: Any, top_k: int = 10, patient_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Queries image collection. 
        If patient_id given, filter by that patient (longitudinal retrieval).
        """
        emb_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding
        where_clause = {"patient_id": patient_id} if patient_id else None
        
        results = self.image_collection.query(
            query_embeddings=[emb_list],
            n_results=top_k,
            where=where_clause
        )
        
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                meta = results['metadatas'][0][i]
                formatted_results.append({
                    "study_id": results['ids'][0][i],
                    "image_path": meta.get("image_path", ""),
                    "score": results['distances'][0][i],
                    "metadata": meta
                })
                
        return formatted_results

    def query_texts(self, query_embedding: Any, top_k: int = 10, patient_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Queries text collection.
        """
        emb_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding
        where_clause = {"patient_id": patient_id} if patient_id else None
        
        results = self.text_collection.query(
            query_embeddings=[emb_list],
            n_results=top_k,
            where=where_clause
        )
        
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                meta = results['metadatas'][0][i]
                formatted_results.append({
                    "chunk_id": results['ids'][0][i],
                    "chunk_text": meta.get("chunk_text", ""),
                    "score": results['distances'][0][i],
                    "metadata": meta
                })
                
        return formatted_results

    def query_temporal(self, query_embedding: Any, patient_id: str, before_date: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Filter by patient_id AND study_date < before_date.
        Search both collections.
        Return combined results sorted by date.
        """
        emb_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding
        
        # ChromaDB logical AND operator for filtering
        where_clause = {
            "$and": [
                {"patient_id": {"$eq": patient_id}},
                {"study_date": {"$lt": before_date}}
            ]
        }
        
        # 1. Query Images
        img_results = self.image_collection.query(query_embeddings=[emb_list], n_results=top_k, where=where_clause)
        
        # 2. Query Texts
        txt_results = self.text_collection.query(query_embeddings=[emb_list], n_results=top_k, where=where_clause)
        
        combined_results = []
        
        if img_results['ids'] and img_results['ids'][0]:
            for i in range(len(img_results['ids'][0])):
                meta = img_results['metadatas'][0][i]
                combined_results.append({
                    "id": img_results['ids'][0][i],
                    "type": "image",
                    "score": img_results['distances'][0][i],
                    "study_date": meta.get("study_date", ""),
                    "metadata": meta
                })
                
        if txt_results['ids'] and txt_results['ids'][0]:
            for i in range(len(txt_results['ids'][0])):
                meta = txt_results['metadatas'][0][i]
                combined_results.append({
                    "id": txt_results['ids'][0][i],
                    "type": "text",
                    "score": txt_results['distances'][0][i],
                    "study_date": meta.get("study_date", ""),
                    "metadata": meta
                })
                
        # Sort combined results chronologically descending (most recent history first)
        # Using string comparison for YYYY-MM-DD
        combined_results.sort(key=lambda x: x["study_date"], reverse=True)
        
        return combined_results
