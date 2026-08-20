# MedVision Channel for HiperHealth

The MedVision channel (`hph-medvision-channel`) is a specialized expansion for the [HiperHealth](https://github.com/hiperhealth/hiperhealth) clinical AI platform.

It provides state-of-the-art medical computer vision capabilities that plug directly into HiperHealth's StageRunner pipeline, enabling multimodal (vision + text) clinical workflows.

## Features

- **Skin Lesion Analysis (`medvision.skin_analysis`)**: An automated intake skill that processes dermoscopic images using a fine-tuned DINOv2 vision transformer. It classifies lesions into 7 categories (e.g., Melanoma, Basal Cell Carcinoma, Benign Nevi).
- **Explainable AI (Grad-CAM)**: Generates heatmaps to highlight the regions of the image that contributed most to the model's prediction, aiding clinician trust and interpretability.
- **Confidence Calibration**: Uses Temperature Scaling to calibrate output probabilities and support clinician-review flagging when confidence is low.
- **Pipeline Integration**: Automatically intercepts the `intake` stage to inject visual observations and injects natural language summaries into the `diagnosis` stage via `prompt_fragments`.

## Installation

This channel is designed to be dynamically loaded by HiperHealth's `SkillRegistry`.

1. Ensure you have HiperHealth installed.
2. Clone this repository:
   ```bash
   git clone https://github.com/hiperhealth/hph-medvision-channel.git
   cd hph-medvision-channel
   ```
3. Install the channel dependencies (requires PyTorch and MONAI):
   ```bash
   conda env create -f conda/dev.yaml
   conda activate medvision-channel
   pip install -e .
   ```
4. Register the channel in your HiperHealth application:

   ```python
   from hiperhealth.pipeline import SkillRegistry

   registry = SkillRegistry()
   registry.add_channel("/path/to/hph-medvision-channel")
   registry.install_skill("medvision.skin_analysis")
   ```

## Usage

Once installed, the `medvision.skin_analysis` skill will automatically execute during the `Stage.INTAKE` hook whenever a `skin_image` is present in the `PipelineContext.patient` data.

```python
from hiperhealth.pipeline import StageRunner, Stage, PipelineContext

ctx = PipelineContext(
    patient={"skin_image": "/path/to/lesion.jpg"}
)

runner = StageRunner(skills=registry.get_active_skills())
ctx = runner.run(Stage.INTAKE, ctx)

# Visual observations are automatically appended to results
print(ctx.results["intake"]["visual_observations"])
```

## Documentation

- [Development Guide](DEVELOPMENT.md): Instructions for setting up the local environment and contributing.
- [Architecture](docs/architecture.md): Deep dive into the MedVision component design and pipeline integration.
- [Code of Conduct](CODE_OF_CONDUCT.md): Community guidelines.

## License

This project is licensed under the BSD Clause 3 License - see the LICENSE file for details.
