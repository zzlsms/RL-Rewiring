# Subspace-Aligned Rewiring (SAR)

Official implementation of **Spectral Rewiring for Exploration, Purification,
and Model Merging**. SAR extracts a compact, reasoning-effective component from
a full-parameter RL update by projecting it into the base model's spectral
space.

- [Paper](https://arxiv.org/abs/2607.03065)
- Checkpoint: Coming soon

## Installation

```bash
git clone https://github.com/zzlsms/RL-Rewiring.git
cd RL-Rewiring

conda create -n sar python=3.11 -y
conda activate sar
pip install -r requirements.txt
```

Tested with Python 3.11.14, PyTorch 2.9.0, Transformers 4.57.3, vLLM 0.12.0,
and CUDA 12.x.

## Checkpoint

Once released, replace the placeholder with the Hugging Face model id:

```bash
export SAR_MODEL=YOUR_HF_NAMESPACE/sar-checkpoint
```


## SAR Projection

Build a SAR checkpoint from architecture-compatible base and RL models:

```bash
BASE_MODEL=BASE_MODEL_ID \
RL_MODEL=RL_MODEL_ID \
SAVE_PATH=checkpoints/sar-checkpoint \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DELTA_FRACTION=0.01 \
bash scripts/run_sar_projection.sh
```

`DELTA_FRACTION` controls the retained fraction of the RL-update rank. The
projection uses float32 matrix SVD and may require substantial CPU RAM and GPU
memory.

## AIME Evaluation

The launcher starts vLLM servers, shards AIME 2024 across workers, and samples
`K_VALUE` responses per problem:

```bash
MODEL_PATH="$SAR_MODEL" \
OUT_DIR=outputs/aime24_sar \
TOTAL_NODES=1 \
NODE_RANK=0 \
GPUS_PER_NODE=4 \
TP_SIZE=1 \
K_VALUE=32 \
bash scripts/run_aime.sh
```

Use `TP_SIZE=4` when one model instance requires four GPUs. For multi-node
evaluation, run the command on every node with the same `TOTAL_NODES`, a unique
zero-based `NODE_RANK`, and a shared `OUT_DIR`.

Aggregate the shard files and report Pass@1, Pass@k, and average generation
length:

```bash
python scripts/summarize_aime24.py \
  --result_dir outputs/aime24_sar \
  --target_k 32 \
  --output outputs/aime24_sar/summary.json
```

The summary script first combines repeated samples of the same problem across
shards, then applies the standard unbiased Pass@k estimator. Evaluation logs
are written to `OUT_DIR/logs/`.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Citation

```bibtex
@article{zhang2026spectralrewiring,
  title   = {Spectral Rewiring for Exploration, Purification, and Model Merging},
  author  = {Zhang, Zhilong and Yu, Hongli and Gao, Huan-ang and Wu, Hanlin and Song, Yuxuan and Ma, Wei-Ying and Zhang, Ya-Qin and Zhou, Hao},
  journal = {arXiv preprint arXiv:2607.03065},
  year    = {2026}
}
```

## License

This code is released under the [MIT License](LICENSE). The model checkpoint
will follow the license requirements of its base model.
