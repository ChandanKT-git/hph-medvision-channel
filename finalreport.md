# GSoC 2026 Final Report: MedVision + Explainability: Medical Image Analysis with Interpretable AI

| | |
|---|---|
| **Contributor** | Chandan K T ([@ChandanKT-git](https://github.com/ChandanKT-git)) |
| **Organization** | [Open Science Labs (OSL)](https://opensciencelabs.org/) |
| **Project** | HiperHealth - MedVision + Explainability |
| **Mentors** | Ivan Ogasawara ([@xmnlab](https://github.com/xmnlab)), Aniket Kumar ([@whitewolf2000ani](https://github.com/whitewolf2000ani)) |
| **Project Size** | Large (~350 hours) |
| **Repository** | [`hiperhealth/hph-medvision-channel`](https://github.com/hiperhealth/hph-medvision-channel) |

---

## Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture](#-architecture)
3. [Work Completed](#-work-completed)
4. [Pull Requests](#-pull-requests)
5. [Blog Posts](#-blog-posts)
6. [Model Training Results](#-model-training-results)
7. [Difficulties & Lessons Learned](#-difficulties--lessons-learned)
8. [Future Scope](#-future-scope)
9. [Acknowledgements](#-acknowledgements)

---

## Project Overview

### The Problem

[HiperHealth](https://github.com/hiperhealth/hiperhealth) is an open-source python library with powerful **text-based clinical workflows** built on a skill-based pipeline architecture. However, **visual diagnostic information**, one of the most valuable data sources in clinical practice was entirely absent. Visual symptoms like skin lesions, nail abnormalities, and eye conditions often provide critical diagnostic clues that text alone cannot capture.

### The Solution

I built **MedVision** a modular, Git-hosted channel of **medical image analysis skills** that plug directly into the existing HiperHealth pipeline. Each skill uses:

- **MONAI + PyTorch** for medical image processing and classification
- **DINOv2 ViT-S/14** (Meta AI) as the vision backbone
- **Grad-CAM** for explainable AI, visual heatmaps showing *what the model sees*
- **Temperature Scaling** for calibrated confidence scores
- **Parquet** for clinical data persistence (via HiperHealth's session system)
- **Zero-modification integration** with the existing `DiagnosticsSkill` via `prompt_fragments`

The key design principle: **visual intelligence becomes a natural part of clinical workflows**, not a separate tool.

### Scope

Rather than building 6 shallow prototypes, I focused on **depth and production quality**:

-  **1 complete, production-quality skill** (SkinAnalysisSkill)
-  **Full shared infrastructure** reusable by any future skill
-  **End-to-end pipeline integration** with comprehensive test suite
-  **DINOv2 model training pipeline** with iterative refinements

---

## Architecture

### High-Level System Design

```
┌──────────────────────────────────────────────────────────────────────┐
│                        HiperHealth Pipeline                          │
│                                                                      │
│  Session (Parquet)  ──→  PipelineContext  ──→  StageRunner            │
│                                                                      │
│  StageRunner runs stages with registered skills:                     │
│                                                                      │
│  SCREENING:   PrivacySkill                                           │
│  INTAKE:      ExtractionSkill, SkinAnalysisSkill ◄── NEW (MedVision) │
│  DIAGNOSIS:   SkinAnalysisSkill (pre), DiagnosticsSkill              │
│  EXAM:        DiagnosticsSkill                                       │
│                                                                      │
│  MedVision skills are loaded via:                                    │
│  SkillRegistry.add_channel("hph-medvision-channel")                  │
│  SkillRegistry.install_skill("mv.skin_analysis")                     │
│  StageRunner.register("mv.skin_analysis")                            │
└──────────────────────────────────────────────────────────────────────┘
```

### MedVision Internal Pipeline

```
Patient uploads skin image
    │
    ▼
ImagePreprocessor.validate()
    → EXIF correction, resolution check, blur detection (Laplacian variance)
    │
    ▼
ImagePreprocessor.preprocess()
    → MONAI transforms: EnsureChannelFirst → Resize(224) → ScaleIntensity → NormalizeIntensity
    │
    ▼
ModelRegistry.get_weights()
    → SHA-256 verified, file-locked, atomic download + cache
    │
    ▼
DINOv2 ViT-S/14 Inference
    → Fine-tuned on ISIC 2018 (10k images, 7 skin lesion classes)
    │
    ├──→ TemperatureScaling(T=1.35) → Calibrated probabilities
    │
    └──→ GradCAMExplainer.generate() → Diagnostic heatmap overlay
    │
    ▼
Structured Observation → ctx.results["intake"]["visual_observations"]
    │
    ▼
SkinAnalysisSkill.pre("diagnosis")
    → Injects findings into ctx.extras["prompt_fragments"]["diagnosis"]
    │
    ▼
DiagnosticsSkill.execute("diagnosis")
    → LLM reads prompt_fragments automatically (zero modifications needed)
    → Returns enriched differential diagnosis with visual findings
```

### Repository Structure

```
hph-medvision-channel/
├── skills-channel.yaml           # Channel manifest (name: medvision, alias: mv)
├── shared/                       # Reusable infrastructure for all skills
│   ├── __init__.py               # Public API exports
│   ├── preprocessing.py          # MONAI transforms, quality validation, EXIF
│   ├── explainability.py         # Grad-CAM, Attention Rollout, overlay utilities
│   ├── confidence.py             # Temperature scaling calibration
│   └── models/
│       └── registry.py           # Download, cache, SHA-256 verify model weights
├── skills/
│   └── skin_analysis/
│       ├── skill.yaml            # Skill manifest (stages: intake, diagnosis)
│       ├── skill.py              # SkinAnalysisSkill(BaseSkill) — all 3 hooks
│       └── labels.json           # SNOMED CT mappings for 7 lesion classes
├── tests/                        # 168 tests, 94% coverage
│   ├── test_preprocessing.py
│   ├── test_explainability.py
│   ├── test_confidence.py
│   ├── test_model_registry.py
│   ├── test_skin_analysis.py
│   ├── test_integration.py
│   └── ...
├── docs/
│   └── architecture.md
└── .github/                      # CI/CD workflows
```

---

## Work Completed

### Phase 1: Infrastructure

| Deliverable | Description |
|---|---|
| **CI/CD Scaffold** | GitHub Actions, pre-commit (ruff, mypy, bandit, vulture), semantic-release, Conda environment, makim task runner |
| **Image Preprocessing** | `ImagePreprocessor` with quality validation (resolution, aspect ratio, blur via Laplacian), EXIF orientation correction, MONAI transform pipeline. 28 unit tests, 92% coverage |
| **Model Registry** | `ModelRegistry` with SHA-256 integrity verification, atomic `.part` downloads, `filelock` concurrency, JSON manifest tracking |
| **Grad-CAM Explainability** | `GradCAMExplainer` supporting both CNN and ViT architectures, `AttentionRollout` fallback for gradient-free scenarios, `reshape_transform` for ViT patch tokens, `overlay_heatmap` visualization, `get_explainer` factory |
| **Confidence Calibration** | `TemperatureScaling` module using L-BFGS optimization to learn optimal temperature on validation logits. Makes 90% confidence *actually* mean 90% |
| **SNOMED CT Labels** | Machine-readable label mapping for HAM10000 dataset 7 skin lesion classes with SNOMED CT codes |

### Phase 2: Model Training

| Deliverable | Description |
|---|---|
| **Exploratory Data Analysis** | Comprehensive EDA on HAM10000 dataset revealing 58:1 class imbalance, 25% duplicate lesions, source-diagnosis correlation. EDA directly shaped every training decision |
| **Training Pipeline** | Complete 5-fold cross-validation pipeline: `config.py`, `dataset.py`, `model.py`, `trainer.py`, `evaluate.py`, `train.py` |
| **DINOv2 Fine-Tuning** | Two-phase transfer learning (Linear Probing → Fine-tuning last 4 blocks). 3 iterative training runs with progressive improvements |
| **Class Imbalance Handling** | Three-layered approach: WeightedRandomSampler + Focal Loss (γ=2.0) + EDA-derived class weights |
| **Data Leakage Prevention** | `StratifiedGroupKFold` grouped by `lesion_id` to prevent same-lesion contamination across folds |
| **Advanced Techniques** | Mixup augmentation, Test-Time Augmentation (5 flips), expanded classification head (384 → 128 → 7) |

### Phase 3: Skill Implementation & Integration

| Deliverable | Description |
|---|---|
| **SkinAnalysisSkill** | Complete `BaseSkill` implementation with 3 lifecycle hooks: `check_requirements` (requests skin image), `execute` (inference + Grad-CAM + calibration), `pre` (injects prompt_fragments) |
| **Safety Mechanisms** | `_determine_requires_review()` auto-flags high-risk findings (melanoma, low confidence) for clinician review |
| **Integration Tests** | End-to-end tests verifying `StageRunner` hook execution order, `PipelineContext` data flow, prompt fragment injection, `Session` persistence |
| **Documentation** | `README.md`, `DEVELOPMENT.md`, `CODE_OF_CONDUCT.md`, `docs/architecture.md` |
| **Test Suite** | 168 unit + integration tests with **94% code coverage** |

---

## Pull Requests

### Upstream Repository ([`hiperhealth/hph-medvision-channel`](https://github.com/hiperhealth/hph-medvision-channel))

| # | PR | Description | Status |
|---|---|---|---|
| 1 | [#2 — `build: setup initial repository and CI/CD scaffold`](https://github.com/hiperhealth/hph-medvision-channel/pull/2) | Established CI/CD scaffolding: `pyproject.toml`, Conda env, pre-commit hooks, GitHub Actions, semantic-release, makim task runner | ✅ **Merged** |
| 2 | [#4 — `feat(preprocessing): add image preprocessing with quality validation`](https://github.com/hiperhealth/hph-medvision-channel/pull/4) | MONAI-based preprocessing pipeline with quality validation (resolution, blur, EXIF), 28 tests with 92% coverage | ✅ **Merged** |
| 3 | [#6 — `feat(model-registry): add ModelRegistry to manage pre-trained weights`](https://github.com/hiperhealth/hph-medvision-channel/pull/6) | Model weight download/cache system with SHA-256 verification, atomic downloads, file locking, JSON manifest | ✅ **Merged** |
| 4 | [#7 — `feat(explainability): add TransformerCAM and AttentionRollout`](https://github.com/hiperhealth/hph-medvision-channel/pull/7) | Grad-CAM for CNN + ViT, AttentionRollout fallback, reshape_transform, overlay_heatmap, get_explainer factory | ✅ **Merged** |
| 5 | [#9 — `feat(shared): add confidence calibration and skin analysis SNOMED labels`](https://github.com/hiperhealth/hph-medvision-channel/pull/9) | Temperature scaling (L-BFGS), SNOMED CT label mappings for 7 skin lesion classes, 100% module coverage | 🔍 **Under Review** |
| 6 | [#10 — `feat(training): DINOv2 ViT-S/14 training pipeline with EDA-driven design`](https://github.com/hiperhealth/hph-medvision-channel/pull/10) | Full training pipeline: EDA, two-phase fine-tuning, Focal Loss, StratifiedGroupKFold, WeightedRandomSampler | 🔍 **Under Review** |
| 7 | [#11 — `docs: add Iteration 3 evaluation logs and visualizations`](https://github.com/hiperhealth/hph-medvision-channel/pull/11) | Training results after 3 iterations: 76.6% balanced accuracy, 0.942 AUC-ROC, per-class metrics breakdown | 🔍 **Under Review** |
| 8 | [#13 — `feat: implement SkinAnalysisSkill`](https://github.com/hiperhealth/hph-medvision-channel/pull/13) | Core skill with DINOv2 inference, Grad-CAM overlays, calibrated confidence, prompt_fragment injection, safety flags, 92.7% coverage | 🔍 **Under Review** |
| 9 | [#14 — `feat(pipeline): add integration tests and project documentation`](https://github.com/hiperhealth/hph-medvision-channel/pull/14) | End-to-end integration tests, README, DEVELOPMENT.md, architecture docs, 94% coverage | 🔍 **Under Review** |

> **Summary: 4 merged, 5 under review** all PRs authored by me during GSoC 2026.

---

## Blog Posts

| # | Title | Link |
|---|---|---|
| 1 | **Introducing MedVision: Bringing Explainable Medical Image Analysis to HiPerHealth** | [Blog Post 1]() <!-- TODO: Add published Medium URL --> |
| 2 | **Building the Foundation: How MedVision Turns a Medical Image into a Trusted Clinical Observation** | [Blog Post 2]() <!-- TODO: Add published Medium URL --> |

**Blog 1** introduces MedVision, explains the problem space, HiPerHealth's skill architecture, and why MONAI and Grad-CAM were chosen as core technologies.

**Blog 2** dives deep into the shared infrastructure: image preprocessing with quality validation, the model registry, Grad-CAM explainability (including the ViT challenge), and confidence calibration via temperature scaling.

---

## Model Training Results

The DINOv2 ViT-S/14 backbone was fine-tuned on the ISIC 2018 / HAM10000 dataset (10,015 images, 7 skin lesion classes) through iterative runs, with each iteration informed by analysis of the previous run's failure modes.

### Final Results (5-Fold Cross-Validation)

| Metric | Result (5-Fold Average) | Best Single Fold |
|---|---|---|
| **Balanced Accuracy** | 76.6% ± 2.4% | **81.1%** (Fold 4) |
| **AUC-ROC (OvR)** | 0.942 ± 0.007 | 0.955 |
| **Macro F1** | 0.580 ± 0.026 | 0.631 |
| **Cohen's Kappa** | 0.503 ± 0.025 | 0.536 |

> Both the AUC-ROC target (≥ 0.90) and the Balanced Accuracy target (≥ 75%) were exceeded. Fold 4 demonstrates the architecture can break 80% accuracy with a favorable data split.

### Key Training Techniques

| Technique | Purpose |
|---|---|
| **Two-Phase Transfer Learning** | Phase 1: Linear probing (15 epochs, freeze backbone). Phase 2: Fine-tune last 4 transformer blocks (25 epochs) |
| **Focal Loss (γ=2.0)** | Down-weights easy predictions, forces model to focus on hard/rare classes |
| **Mixup Augmentation** | Blends image-label pairs to reduce overconfidence, significantly improved AUC-ROC |
| **Test-Time Augmentation** | Evaluates 5 flipped variants at inference, averages logits for more robust predictions |
| **StratifiedGroupKFold** | Groups by `lesion_id` to prevent data leakage from duplicate photographs of the same lesion |
| **EDA-Derived Class Weights** | Computed from dataset analysis (58:1 imbalance), ranges from 0.21 (nv) to 12.44 (df) |


## Difficulties & Lessons Learned

### 1. Extreme Class Imbalance (58:1)

The HAM10000 dataset has a severe imbalance, `melanocytic_nevi` (6,705 images) outnumbers `dermatofibroma` (115 images) by a factor of 58. Standard training approaches resulted in the model predicting the majority class almost exclusively.

**Solution:** A three-layered strategy (WeightedRandomSampler + Focal Loss + EDA-derived class weights) was needed to achieve balanced performance across all classes. No single technique was sufficient alone.

### 2. Data Leakage from Duplicate Lesions

The EDA revealed that ~25% of images are re-photographs of the same physical lesion. Using standard `StratifiedKFold` caused the same lesion to leak into both training and validation sets, inflating metrics by ~5-8%.

**Solution:** Switched to `StratifiedGroupKFold` with grouping by `lesion_id`. This required careful analysis of the metadata and significantly changed the validation strategy.

### 3. Grad-CAM for Vision Transformers

Most Grad-CAM implementations assume CNNs with 2D spatial feature maps. DINOv2's ViT produces flat token sequences, breaking standard Grad-CAM. The CLS token (position 0) has no spatial meaning and must be excluded.

**Solution:** Implemented a `reshape_transform` function that strips the CLS token, reshapes remaining patch tokens into a 2D grid (16×16 for ViT-S/14 with 224×224 input), and transposes to the expected (B, C, H, W) format. Also implemented `AttentionRollout` as a gradient-free fallback.

### 4. EXIF Orientation in Clinical Photos

Phone cameras store photos in sensor orientation with an EXIF tag for display rotation. OpenCV's `imread` ignores this tag, causing portrait photos to arrive rotated 90°. This is invisible in most image viewers but produces wrong model predictions.

**Solution:** Added `correct_exif_orientation()` that reads the EXIF `Orientation` tag via PIL and applies the corresponding OpenCV rotation before any processing.

### General Lessons

- **Medical imaging ≠ computer vision with different images.** Quality constraints, failure modes, and downstream consumers are fundamentally different. Every design decision was shaped by clinical context.
- **Build infrastructure before skills.** The shared modules (preprocessing, explainability, calibration, registry) took the most time but made the SkinAnalysisSkill implementation straightforward.
- **Explainability cannot be an afterthought.** Designing for Grad-CAM from the start avoided architectural conflicts that would arise from bolting it on later.
- **Open source means obvious code.** Every function has a docstring, every error has a structured detail dict, every module has a test suite because future contributors need to understand the system without reading the implementation.

---

## Future Scope

### Short-Term (Post-GSoC)

| Goal | Description |
|---|---|
| **Hosting Model** | Hosting model on a server to experiment with quick inferences |
| **Second body-region skill** | Eye analysis (ODIR dataset, 5k images, 8 classes) or Nail analysis validates architecture generalizability using the same shared infrastructure |
| **Bias & Fairness Audit** | Evaluate model performance across Fitzpatrick Skin Types I–VI. Flag and mitigate if accuracy disparity exceeds 15% between groups |

### Long-Term

| Goal | Description |
|---|---|
| **NailAnalysisSkill** | Nail abnormalities (clubbing, Beau's lines, melanonychia) as systemic disease indicators |
| **TongueAnalysisSkill** | Tongue coating, glossitis, geographic tongue assessment |
| **EarAnalysisSkill** | Otoscopic image analysis |
| **NutritionEstimatorSkill** | Food recognition from meal photographs for dietary assessment |
| **SHAP Integration** | Feature importance analysis alongside Grad-CAM heatmaps |

---

## 🙏 Acknowledgements

This project would not have been possible without the guidance and support of:

- **Ivan Ogasawara** ([@xmnlab](https://github.com/xmnlab)) for architecting HiperHealth's elegant pipeline system and providing mentorship throughout the project
- **Aniket Kumar** ([@whitewolf2000ani](https://github.com/whitewolf2000ani)) for thorough code reviews and feedback that significantly improved code quality
- **Open Science Labs** for creating an environment where open-source healthcare technology can thrive
- **Google Summer of Code**  for the opportunity to contribute meaningfully to open-source healthcare AI

---

## Links

| Resource | URL |
|---|---|
| **Main Repository** | https://github.com/hiperhealth/hph-medvision-channel |
| **Fork** | https://github.com/ChandanKT-git/hph-medvision-channel |
| **HiperHealth Core** | https://github.com/hiperhealth/hiperhealth |
| **HiperHealth Documentation** | https://hiperhealth.github.io/hiperhealth/ |
| **Organization** | https://opensciencelabs.org/ |
| **Blog Post 1** | <!-- TODO: Add Medium URL --> |
| **Blog Post 2** | <!-- TODO: Add Medium URL --> |
