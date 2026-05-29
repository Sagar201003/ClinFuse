import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import joblib
import hashlib
from pathlib import Path

class TextEncoder:
    def __init__(self, cache_dir="cache/text"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading SapBERT Text Encoder on {self.device}...")
        self.model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print("SapBERT Text Encoder loaded successfully.")

    def _get_cache_path(self, text: str) -> Path:
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return self.cache_dir / f"{text_hash}_emb.joblib"

    def _mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on token embeddings.
        """
        token_embeddings = model_output[0] # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encodes text using SapBERT. Handles texts > 512 tokens by chunking.
        Returns a mean-pooled, L2-normalized embedding.
        """
        cache_path = self._get_cache_path(text)
        if cache_path.exists():
            return joblib.load(cache_path)

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        max_length = 510 # Reserve 2 tokens for CLS and SEP
        
        # Chunk the tokens if they exceed max length
        chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
        chunk_embeddings = []
        
        with torch.no_grad():
            for chunk in chunks:
                inputs = self.tokenizer.prepare_for_model(
                    chunk, 
                    add_special_tokens=True, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                ).to(self.device)
                
                outputs = self.model(**inputs)
                
                # Mean pool token embeddings
                chunk_emb = self._mean_pooling(outputs, inputs['attention_mask'])
                chunk_embeddings.append(chunk_emb)
                
        # Average the chunks if there are multiple
        if chunk_embeddings:
            final_embedding = torch.mean(torch.stack(chunk_embeddings), dim=0)
        else:
            final_embedding = torch.zeros((1, 768)).to(self.device)
            
        # Normalize to unit length (L2 norm)
        final_embedding = torch.nn.functional.normalize(final_embedding, p=2, dim=1)
        emb_np = final_embedding.cpu().numpy().squeeze()
        
        joblib.dump(emb_np, cache_path)
        return emb_np

    def encode_texts(self, texts: list) -> np.ndarray:
        """
        Batch version of text encoding.
        """
        embeddings = [self.encode_text(text) for text in texts]
        return np.vstack(embeddings)

if __name__ == "__main__":
    print("Testing Text Encoder...")
    encoder = TextEncoder()
    
    sample_text = "Patient presents with acute chest pain and shortness of breath. History of hypertension."
    emb = encoder.encode_text(sample_text)
    print(f"Encoded single text. Shape: {emb.shape}")
    
    # Test long text chunking (creating a long string)
    long_text = "chest pain " * 600 
    long_emb = encoder.encode_text(long_text)
    print(f"Encoded long text (>512 tokens). Shape: {long_emb.shape}")
    
    embs = encoder.encode_texts([sample_text, long_text])
    print(f"Encoded batch texts. Shape: {embs.shape}")
