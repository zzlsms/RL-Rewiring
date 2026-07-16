---
library_name: transformers
pipeline_tag: text-generation
license: mit
base_model: BASE_MODEL_ID
tags:
  - reasoning
  - model-merging
  - reinforcement-learning
---

# SAR Checkpoint

This repository contains a checkpoint released with **Spectral Rewiring for
Exploration, Purification, and Model Merging**.

## Model Details

- Base model: `BASE_MODEL_ID`
- RL model: `RL_MODEL_ID`
- Method: Subspace-Aligned Rewiring (SAR)
- SAR delta fraction: `DELTA_FRACTION`

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "YOUR_HF_NAMESPACE/sar-checkpoint"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
)
```

## Evaluation

See the GitHub repository accompanying the paper for the released evaluation
scripts and exact commands.

## Limitations

This checkpoint inherits the limitations and usage restrictions of its base
model. Users should independently validate outputs in high-stakes settings.

## Citation

```bibtex
@article{zhang2026spectralrewiring,
  title   = {Spectral Rewiring for Exploration, Purification, and Model Merging},
  author  = {Zhang, Zhilong and Yu, Hongli and Gao, Huan-ang and Wu, Hanlin and Song, Yuxuan and Ma, Wei-Ying and Zhang, Ya-Qin and Zhou, Hao},
  journal = {arXiv preprint arXiv:2607.03065},
  year    = {2026}
}
```
