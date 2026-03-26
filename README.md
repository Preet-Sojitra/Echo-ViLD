# Echo-ViLD: Segment-Guided Audio-Visual Distillation for Zero-Shot Object Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

This repository contains the official implementation for **Echo-ViLD**, a research-oriented course project for CS 6384 (Computer Vision) at The University of Texas at Dallas.

## Project Overview

Open-Vocabulary Object Detection (OVD) models, such as [ViLD](https://arxiv.org/abs/2104.13921), successfully distill knowledge from vision-language foundation models (e.g., CLIP) to detect novel categories in a zero-shot manner. However, existing methodologies face two critical limitations:
1. **Visual Cropping Artifacts:** Distilling from rectangular bounding boxes introduces noisy background pixels and distorts aspect ratios, degrading the quality of the teacher's target embeddings.
2. **Uni-Modal Inference:** Current OVD models rely exclusively on text queries, ignoring the rich, omni-directional modality of audio for object localization.

**Echo-ViLD** addresses these limitations by introducing a novel, tri-modal distillation framework. We replace CLIP with Meta's **PEAV (Perception Audio-Visual-Text)** foundation model and introduce an offline **Segment-Guided Distillation** pipeline using the Segment Anything Model (SAM) to generate high-fidelity, background-free target embeddings. 

By mapping our student detector into PEAV's unified latent space, Echo-ViLD enables true **Zero-Shot Acoustic Object Localization**, allowing users to detect sounding objects using `.wav` audio queries.

---

## Key Features & Novelties

*   **Segment-Guided Distillation:** Utilizing SAM to mask background noise from RPN proposals before teacher embedding generation.
*   **Multiscale Contextual Fusion:** Fusing SAM-cleaned object embeddings with global scene context to resolve semantic ambiguity.
*   **Audio-Driven Zero-Shot Inference:** Inheriting PEAV's tri-modal latent space to enable bounding-box detection via audio clips (e.g., VGGSound).
*   **Dense LLM Prompting:** Replacing static text templates with rich, LLM-generated visual descriptions to improve cross-entropy text distillation.

---

## Project Architecture

> To be added later


## Evaluation Metrics

> To be added later

## Repository Structure

> To be added later

## Datasets

> To be added later 

## Team Members

> To add

## References

> To add