# ClinFuse: Multimodal RAG for Clinical Decision Support 🏥

**Work in progress 🚧**

## The Problem Statement
Current LLM-based clinical assistants often hallucinate or rely solely on text (such as EHR notes and discharge summaries). They fail to jointly reason over visual evidence (like X-rays or histology slides) and patient text records simultaneously. This lack of multimodal integration makes them unreliable for real, reliable diagnostic support where visual findings and patient history must be evaluated together.

## The Solution: ClinFuse
ClinFuse is a Multimodal Retrieval-Augmented Generation (RAG) architecture designed to ground LLM responses in both **Medical Imaging** and **EHR Text**. 

By embedding both text and images into a shared semantic space, employing advanced retrieval strategies (like Reciprocal Rank Fusion and Temporal context retrieval), and scoring claims based on multi-modal evidence, ClinFuse aims to provide robust, grounded clinical decision support.

## Current Pipeline Components

* **Data Pipeline**: Tools for loading MIMIC-CXR datasets, chunking EHR text, and building chronological patient timelines. Includes a specialized `hf_unpacker.py` to seamlessly adapt HuggingFace Parquet datasets into the raw PhysioNet folder structure.
* **Vision & Text Encoders**: Embeds Chest X-rays using `BioViL-T` and medical texts using `SapBERT`.
* **HyDE Module (Novelty 3)**: Leverages 4-bit quantized LLaMA 3.1 8B to generate hypothetical radiology reports from symptoms, bridging the modality gap for accurate image retrieval.
* **Retrieval Infrastructure**: Persistent `ChromaDB` handling dual text/image collections, paired with sparse `BM25` retrieval over clinical notes.
* **Cross-Modal RRF Fusion & Reranking (Novelty 1)**: Intelligently merges Dense Image, Dense Text, and Sparse Text hits via Reciprocal Rank Fusion, followed by precision scoring using a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`).
* **Temporal Retrieval (Novelty 2)**: Longitudinal RAG engine that fetches current clinical context alongside chronologically accurate historical records for the same patient.
* **Confidence Scorer & Hallucination Prevention**: Splits LLM generations into discrete claims, scoring each sentence against the retrieved multi-modal evidence to calculate grounding thresholds and flag high-risk hallucinations.
