# CEREBRUM: TRIBE-Based Brain Engagement Analysis

Neural engagement analysis platform using Meta's TRIBE v2 model to measure how videos activate different brain regions.

## Pipeline Overview

1. **Data:** Video stimuli are ingested and prepared.
2. **Inference:** TRIBE v2 model extracts predicted neural responses frames.
3. **ROI:** Region-Of-Interest activation mapping is applied to the predictions.
4. **Insights:** Summarization and insights generation from the brain activity.

## Requirements

You can install all necessary dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
```

## How to Run

You can run the pipeline locally (CPU) or on Google Colab.

### Local (CPU) & Colab

Run the pipeline script to process videos. Here's an example command to perform a dry-run on a single video without ablations:

```bash
python run_pipeline.py --dry-run 1 --no-ablation
```

> **Note about model weights:**
> TRIBE v2 weights are not included in this repository. Place `best.ckpt` in the `models/` directory before running the pipeline.
