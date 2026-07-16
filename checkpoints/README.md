# Checkpoints

This directory is reserved for optional local copies of SAR-related
checkpoints. Its model files are excluded by `.gitignore` and must not be
committed to GitHub.

Recommended layout:

```text
checkpoints/
├── base-model/
├── full-rl-model/
└── sar-projected-model/
```

You can also skip local checkpoint storage and pass a Hugging Face model id directly:

```bash
python scripts/eval_jsonl.py \
  --model <hf-model-id> \
  --data examples/math_demo.jsonl
```

To download a published checkpoint explicitly:

```bash
hf download YOUR_HF_NAMESPACE/sar-checkpoint \
  --local-dir checkpoints/sar-checkpoint
```

See [Publishing Checkpoints](../docs/publishing_checkpoints.md) for the server
upload and release workflow.
