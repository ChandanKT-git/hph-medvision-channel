# MedVision Channel Architecture

This document outlines the architecture of the MedVision channel and how it integrates with the core HiperHealth platform.

## Overview

The MedVision channel adds multimodal computer vision capabilities to HiperHealth. Its primary purpose is to process clinical images, generate explainable insights, and seamlessly inject these visual observations into the text-based LLM diagnostic pipeline.

## Core Components

The channel is composed of four main pillars:

### 1. `SkinAnalysisSkill`

The central component that implements `hiperhealth.pipeline.BaseSkill`. It orchestrates the entire computer vision workflow and intercepts specific stages of the HiperHealth pipeline:

- **Intake Stage (`execute` hook):**
  - Retrieves the `skin_image` from the patient context.
  - Passes the image through the preprocessing, model inference, and explainability modules.
  - Formats the raw model outputs (logits, heatmaps) into a structured dictionary and saves it to `ctx.results["intake"]["visual_observations"]`.
- **Diagnosis Stage (`pre` hook):**
  - Reads the structured visual observations from the intake results.
  - Formats them into a Markdown string summary.
  - Injects this string into `ctx.extras["prompt_fragments"]["diagnosis"]` to ensure the downstream LLM receives the visual findings before generating its response.

### 2. Deep Learning Models (`shared/models`)

- Uses a **DINOv2 vision transformer** fine-tuned on dermoscopic datasets (e.g., HAM10000).
- Managed by a custom model registry that automatically downloads required `.pth` weights if they are missing locally.
- Infers probabilities for 7 distinct skin lesion classes, mapping directly to SNOMED CT codes for clinical interoperability.

### 3. Explainability (`shared/explainability.py`)

- Implements **Grad-CAM** (Gradient-weighted Class Activation Mapping) to provide visual explanations for model predictions.
- Generates a transparent heatmap overlay on the original lesion image, highlighting the regions that the model focused on.
- Heatmaps are saved to the local temporary directory (`/tmp/medvision/heatmaps/`) and their paths are included in the pipeline context for UI rendering.

### 4. Confidence Calibration (`shared/confidence.py`)

- Employs **Temperature Scaling** to calibrate the raw model logits.
- Ensures that the output probabilities represent true likelihoods (e.g., a prediction with 80% confidence is correct 80% of the time).
- Helps prevent silent failures by explicitly flagging predictions that fall below a certain confidence threshold or are deemed "out of distribution" based on entropy metrics.

## Data Flow Diagram

```mermaid
graph TD
    A[Patient Uploads Image] -->|Session Context| B(PipelineContext)
    B -->|Stage.INTAKE| C(SkinAnalysisSkill.execute)
    C --> D[Preprocessing]
    D --> E[DINOv2 Inference]
    E --> F[Confidence Calibration]
    E --> G[Grad-CAM Heatmap]
    F --> H{Visual Observations dict}
    G --> H
    H -->|Saves to| B
    B -->|Stage.DIAGNOSIS| I(SkinAnalysisSkill.pre)
    I -->|Reads Observations| J[Markdown Formatter]
    J -->|Injects to| K(ctx.extras['prompt_fragments']['diagnosis'])
    K --> L[LLM Diagnostics Engine]
```

## Security & Privacy Considerations

Because this channel processes sensitive Patient Health Information (PHI) in the form of clinical images:

- **No external API calls:** All inference is performed entirely locally on the host machine.
- **Data minimization:** The channel only extracts the `skin_image` field from the patient context.
- **Ephemeral processing:** Heatmaps are generated in temporary directories and are not persisted to the long-term HiperHealth Parquet logs unless explicitly configured.
