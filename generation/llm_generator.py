import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict, Any

class ClinFuseGenerator:
    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", use_mock: bool = False):
        """
        Initializes the LLM Generator with strict hallucination-prevention prompts 
        and efficient 4-bit quantization.
        """
        self.use_mock = use_mock
        
        # Strict, hardcoded system prompt for clinical grounding
        self.system_prompt = (
            "You are ClinFuse, a clinical decision support assistant. "
            "You ONLY use the provided retrieved evidence to answer. "
            "For every claim you make, cite the evidence source as [IMG-1], "
            "[TXT-1], [HIST-1] etc. based on the evidence list provided. "
            "If evidence is insufficient, say \"Insufficient evidence — "
            "recommend further workup.\" Never fabricate clinical information."
        )
        
        if not self.use_mock:
            print("Loading LLaMA 3.1 8B Instruct with 4-bit NF4 quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto"
            )
            print("LLM Loaded successfully.")
        else:
            print("Running in MOCK mode for LLM Generator.")

    def generate(self, query: str, temporal_context: str, top_k_results: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved multi-modal and temporal evidence into explicitly tagged 
        clinical context and generates a grounded response.
        """
        evidence_lines = []
        img_idx = 1
        txt_idx = 1
        hist_idx = 1
        
        # 1. Format the evidence list with exact citation tags
        for res in top_k_results:
            modality = res.get("modality", res.get("type", "text"))
            temporal_label = res.get("temporal_label", "current")
            meta = res.get("metadata", {})
            
            if "historical" in temporal_label.lower():
                # Extract date safely
                date_str = temporal_label.split("-", 1)[1].strip() if "-" in temporal_label else temporal_label
                text_content = meta.get("report_text", meta.get("chunk_text", "No text provided."))
                evidence_lines.append(f"[{'HIST'}-{hist_idx}] Historical Evidence ({date_str}): {text_content}")
                hist_idx += 1
            elif modality == "image":
                text_content = meta.get("report_text", "No text provided.")
                evidence_lines.append(f"[{'IMG'}-{img_idx}] X-ray finding: {text_content}")
                img_idx += 1
            else:
                text_content = res.get("chunk_text", meta.get("chunk_text", "No text provided."))
                evidence_lines.append(f"[{'TXT'}-{txt_idx}] EHR Note: {text_content}")
                txt_idx += 1
                
        evidence_str = "\n".join(evidence_lines)
        
        # Combine any pre-formatted temporal context if provided
        temporal_block = f"\n[PRE-FORMATTED TEMPORAL TIMELINE]\n{temporal_context}\n" if temporal_context else ""
        
        # 2. Construct the strict User prompt
        user_message = (
            f"PATIENT QUERY: {query}\n\n"
            f"RETRIEVED EVIDENCE:\n{evidence_str}\n"
            f"{temporal_block}"
            f"\nBased ONLY on the above evidence, answer the clinical query. "
            f"Cite each claim with its evidence tag."
        )
        
        if self.use_mock:
            # Bypass GPU requirement for fast local testing
            return (
                f"(MOCK LLM RESPONSE) Based on [TXT-1], the patient shows signs of pneumonia. "
                f"This is corroborated by [IMG-1] which shows opacities. "
                f"Compared to [HIST-1], the condition has worsened."
            )

        # 3. Format using LLaMA's chat template
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.model.device)
        
        # 4. Generate with extremely low temperature to minimize hallucinations
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # 5. Extract only the newly generated tokens
        response_ids = outputs[0][input_ids.shape[-1]:]
        decoded = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return decoded
