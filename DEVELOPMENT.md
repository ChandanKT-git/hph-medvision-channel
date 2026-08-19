# Contributor Guide

First off, thank you for considering contributing to the `hph-medvision-channel`! We welcome all contributions, from bug reports to new features and documentation improvements. This guide provides everything you need to get your development environment set up and start contributing.

Following these guidelines helps to communicate that you respect the time of the developers managing and developing this open source project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Project Layout](#project-layout)
- [Getting Started: Local Setup](#1-getting-started-local-setup)
- [Development Workflow](#2-development-workflow)
  - [Code Style & Linting](#code-style--linting)
  - [Running Tests](#running-tests)
- [Types of Contributions](#types-of-contributions)
- [Architectural Overview](#3-architectural-overview)

## Code of Conduct

This project and everyone participating in it is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

---

## Project Layout

This project uses a flat layout to organize the channel's specific components alongside the main HiperHealth repository ecosystem.

```text
hph-medvision-channel/
├── skills/
│   └── skin_analysis/          # MedVision Skin Analysis Skill implementation
├── shared/
│   ├── models/                 # Model registry and loaders
│   ├── preprocessing.py        # Image preprocessing pipelines
│   ├── explainability.py       # Grad-CAM heatmap generators
│   └── confidence.py           # Temperature scaling calibration
├── tests/                      # Unit and integration tests
├── conda/                      # Environment configuration files
├── skills-channel.yaml         # HiperHealth channel manifest
└── pyproject.toml              # Project dependencies and tool configurations
```

---

## 1. Getting Started: Local Setup

This project uses **Conda** to manage environments and **Makim** to streamline development tasks. The underlying dependencies are heavily focused on computer vision (`torch`, `torchvision`, `monai`).

### Tech Stack Overview

As a medical computer vision channel, this repository relies on:

- **Python 3.10+** for core logic.
- **PyTorch & MONAI** for deep learning model inference.
- **OpenCV & Pillow** for image preprocessing.
- **HiperHealth Pipeline** for workflow integration (`PipelineContext`, `StageRunner`, `BaseSkill`).
- **Makim** as the task runner.

### Prerequisites

- Python 3.10+
- Conda installed on your system.

### Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/hiperhealth/hph-medvision-channel.git
    cd hph-medvision-channel
    ```

2.  **Create the Development Environment:** Set up a Conda virtual environment using the provided `conda/dev.yaml` file:

    ```bash
    conda env create -f conda/dev.yaml
    conda activate medvision-channel
    ```

3.  **Install Project Dependencies:** This command installs all required packages, including development tools like `pytest` and `ruff`:

    ```bash
    pip install -e ".[dev]"
    ```

---

## 2. Development Workflow

### Code Style & Linting

We enforce strict coding standards to maintain readability and consistency across the codebase. We use `ruff` as our primary linter and formatter, along with `mypy` for static type checking.

To check your code:

```bash
makim tests.lint
```

If you need to automatically fix formatting issues:

```bash
ruff format .
ruff check --fix .
```

### Running Tests

We use `pytest` for unit and integration testing. Ensure your tests cover new functionality and don't break existing ones. We require a minimum of 90% test coverage.

To run the entire test suite and generate a coverage report:

```bash
makim tests.unit
```

If you need to run specific tests, you can pass arguments to pytest via Makim:

```bash
makim tests.unit --path tests/test_skin_analysis.py
```

---

## 3. Architectural Overview

The `hph-medvision-channel` is designed as a modular expansion to HiperHealth. Its primary entry point is the `skills-channel.yaml` manifest, which is discovered by HiperHealth's `SkillRegistry`.

The core component is the `SkinAnalysisSkill`, which subclasses `hiperhealth.pipeline.BaseSkill`. This skill registers for two primary hooks:

1. **Intake (`execute`):** Performs DINOv2 model inference, Grad-CAM heatmap generation, and confidence calibration, injecting the results into `ctx.results["intake"]["visual_observations"]`.
2. **Diagnosis (`pre`):** Reads the generated visual observations and formats them into a Markdown prompt fragment, injecting it into `ctx.extras["prompt_fragments"]["diagnosis"]` so that the LLM diagnostic engine has access to the visual findings.

For a deeper dive into the computer vision architecture, please see [Architecture](docs/architecture.md).
