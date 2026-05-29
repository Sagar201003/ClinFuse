import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
import numpy as np
import joblib
import os
from pathlib import Path

class VisionEncoder:
    def __init__(self, cache_dir="cache/vision"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading BioViL-T Vision Encoder on {self.device}...")
        self.model_name = "microsoft/BioViL-T"
        
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
        self.model.eval()
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print("BioViL-T Vision Encoder loaded successfully.")

    def _get_cache_path(self, image_path: str) -> Path:
        basename = Path(image_path).stem
        return self.cache_dir / f"{basename}_emb.joblib"

    def encode_image(self, image_path: str) -> np.ndarray:
        """
        Loads an image, processes it through BioViL-T, extracts the CLS token, 
        and normalizes the embedding to unit length.
        """
        cache_path = self._get_cache_path(image_path)
        if cache_path.exists():
            return joblib.load(cache_path)
            
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return np.zeros(768, dtype=np.float32)

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Extract CLS token from last hidden state
            # Handling different possible outputs from AutoModel for Vision architectures
            if hasattr(outputs, "last_hidden_state"):
                cls_embedding = outputs.last_hidden_state[:, 0, :]
            elif hasattr(outputs, "image_embeds"): 
                cls_embedding = outputs.image_embeds
            else:
                # Fallback to first element (usually hidden states) -> batch_idx 0, cls_idx 0
                cls_embedding = outputs[0][:, 0, :]
                
            # Normalize to unit length (L2 norm)
            cls_embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=1)
            emb_np = cls_embedding.cpu().numpy().squeeze()
            
        joblib.dump(emb_np, cache_path)
        return emb_np

    def encode_images(self, image_paths: list) -> np.ndarray:
        """
        Batch version of image encoding.
        """
        embeddings = [self.encode_image(path) for path in image_paths]
        return np.vstack(embeddings)

if __name__ == "__main__":
    print("Testing Vision Encoder...")
    encoder = VisionEncoder()
    
    # Create a dummy image for testing
    dummy_path = "dummy_test_image.jpg"
    Image.new('RGB', (224, 224), color='gray').save(dummy_path)
    
    emb = encoder.encode_image(dummy_path)
    print(f"Encoded single image. Shape: {emb.shape}")
    
    embs = encoder.encode_images([dummy_path, dummy_path])
    print(f"Encoded batch images. Shape: {embs.shape}")
    
    # Cleanup
    if os.path.exists(dummy_path):
        os.remove(dummy_path)
