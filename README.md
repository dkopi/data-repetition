# Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning

This repository contains the code for reproducing the experiments in [Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning](https://arxiv.org/abs/2602.11149).

We show that, under a fixed update budget, training for more epochs on smaller datasets outperforms single-epoch training on larger datasets in long chain-of-thought SFT.

## Setup

We use [uv](https://docs.astral.sh/uv/) for dependency management. To install all dependencies:

```bash
uv sync
```

Alternatively, see `pyproject.toml` for the full list of dependencies (requires Python 3.12).

## Datasets

The datasets used in our experiments are available in the [HuggingFace collection](https://huggingface.co/collections/dakopi/data-repetition).

To recreate the datasets from scratch, the scripts in `data/` can be used:

- `data/distill.py` — distill long-CoT responses from a teacher model on NuminaMath
- `data/verify.py` — verify distilled responses and split into positive/negative
- `data/intersect.py` — intersect datasets by shared sample indices (used in distillation experiments -- so datasets distilled from 0.6B and 8B Qwen3 contain the same problems)
- `data/split.py` — create train/val splits of varying sizes
- `data/convert_dolci.py` — convert the Dolci-Think-SFT dataset
- `data/prepare_gpqa.py` — prepare the GPQA Diamond evaluation set

## Training

`train.py` runs supervised fine-tuning. It supports full fine-tuning and LoRA, logs to W&B, and computes token-level training metrics. Trained model is being saved under `ckpts/<run_id>`.

```bash
python train.py \
    --run_id example \
    --model allenai/Olmo-3-1025-7B \
    --tokenizer allenai/Olmo-3-7B-Instruct \
    --dataset dakopi/dolci_think__train_200 \
    --epochs 128 \
    --lr 2e-5 \
    --wandb
```

See `scripts/` for the full set of experiment configurations used in the paper.

## Evaluation

`eval.py` evaluates a trained checkpoint using vLLM with pass@k and avg@k metrics.

```bash
python eval.py \
    --model ckpts/<run_id> \
    --tokenizer allenai/Olmo-3-7B-Instruct \
    --task aime24
```

Supported tasks: `aime24`, `aime25`, `gpqa`.

## Citation

```bibtex
@misc{kopiczko2026datarepetitionbeatsdata,
      title={Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning}, 
      author={Dawid J. Kopiczko and Sagar Vaze and Tijmen Blankevoort and Yuki M. Asano},
      year={2026},
      eprint={2602.11149},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.11149}, 
}
```
