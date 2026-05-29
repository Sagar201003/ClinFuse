import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the text encoder
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from encoders.text_encoder import TextEncoder

class HyDEGenerator:
    def __init__(self, use_mock=False):
        """
        Initializes the HyDE Generator. 
        use_mock is useful for testing without downloading the 8B model.
        """
        self.use_mock = use_mock
        self.system_prompt = (
            "You are an expert radiologist. Given a clinical symptom description, "
            "write a concise hypothetical chest X-ray radiology report (findings + "
            "impression section only, max 150 words) that would be consistent with "
            "these symptoms. Be clinically precise."
        )
        
        # Load the text encoder for SapBERT embeddings
        self.text_encoder = TextEncoder()
        
        if not self.use_mock:
            print("Loading LLaMA 3.1 8B Instruct in 4-bit NF4...")
            self.model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            
            # 4-bit Quantization Config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    quantization_config=bnb_config,
                    device_map="auto"
                )
                print("LLaMA loaded successfully.")
            except Exception as e:
                print(f"Warning: Failed to load LLaMA ({e}). Falling back to mock generator.")
                self.use_mock = True
        else:
            print("HyDE Initialized in MOCK mode (skipping LLaMA loading).")

    def generate_hypothetical_report(self, symptom_text: str) -> str:
        """
        Generates a hypothetical radiology report based on symptom text using LLaMA.
        """
        if self.use_mock:
            return f"FINDINGS: Bilateral infiltrates consistent with symptoms of '{symptom_text}'.\nIMPRESSION: Suspect infectious or inflammatory process."
            
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Symptoms: {symptom_text}\n\nGenerate the hypothetical radiology report."}
        ]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Extract only the newly generated text (ignoring the prompt)
        response_ids = outputs[0][input_ids.shape[-1]:]
        report = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        return report

    def get_hyde_embedding(self, symptom_text: str) -> np.ndarray:
        """
        Generates the hypothetical report and encodes it with SapBERT.
        """
        print(f"Generating hypothetical report for symptoms: '{symptom_text}'...")
        hypothetical_report = self.generate_hypothetical_report(symptom_text)
        print(f"Embedding the generated report...")
        embedding = self.text_encoder.encode_text(hypothetical_report)
        return embedding

    def hyde_image_retrieval(self, symptom_text: str, image_collection, top_k: int = 5):
        """
        Queries a ChromaDB image collection using the HyDE text embedding.
        """
        hyde_emb = self.get_hyde_embedding(symptom_text)
        
        results = image_collection.query(
            query_embeddings=[hyde_emb.tolist()],
            n_results=top_k
        )
        
        return results

if __name__ == "__main__":
    print("Testing HyDE Module...")
    # Note: Using mock=True for quick testing without GPU/LLaMA overhead
    hyde_gen = HyDEGenerator(use_mock=True)
    
    symptom = "65 year old male, 3 week cough, night sweats, smoker"
    report = hyde_gen.generate_hypothetical_report(symptom)
    
    print("\n" + "="*50)
    print("Hypothetical Report:")
    print("="*50)
    print(report)
    print("="*50 + "\n")
    
    emb = hyde_gen.get_hyde_embedding(symptom)
    print(f"HyDE embedding shape: {emb.shape}")
