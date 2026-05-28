# ClinFuse: Multimodal RAG for Clinical Decision Support 🏥

**Work in progress 🚧**

## The Problem Statement
Current LLM-based clinical assistants often hallucinate or rely solely on text (such as EHR notes and discharge summaries). They fail to jointly reason over visual evidence (like X-rays or histology slides) and patient text records simultaneously. This lack of multimodal integration makes them unreliable for real, reliable diagnostic support where visual findings and patient history must be evaluated together.

## The Solution: ClinFuse
ClinFuse is a Multimodal Retrieval-Augmented Generation (RAG) architecture designed to ground LLM responses in both **Medical Imaging** and **EHR Text**. 

By embedding both text and images into a shared semantic space, employing advanced retrieval strategies (like Reciprocal Rank Fusion and Temporal context retrieval), and scoring claims based on multi-modal evidence, ClinFuse aims to provide robust, grounded clinical decision support.

## Current Pipeline Components
* **Data Pipeline**: Tools for loading MIMIC-CXR datasets, chunking EHR text with overlaps, and building chronological patient timelines for temporal retrieval context.
