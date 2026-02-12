import unsloth
from unsloth import FastLanguageModel
import argparse
from transformers import AutoTokenizer, GenerationConfig
import torch
import torch.nn.functional as F
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
from pathlib import Path


def compute_token_metrics(
    logits: torch.Tensor,  # [B, T, V]
    targets: torch.Tensor,  # [B, T]
):
    logits_2d = logits.view(-1, logits.size(-1))
    targets_1d = targets.view(-1)

    # NLL / Loss Stats
    nll_per_tok = F.cross_entropy(logits_2d, targets_1d, reduction="none")
    loss_std = nll_per_tok.std()
    loss_mean = nll_per_tok.mean()

    # Token Accuracy
    preds = logits_2d.argmax(dim=-1)
    token_acc = (preds == targets_1d).float().mean()

    # Top-K and Margins
    top2_logits, _ = logits_2d.topk(2, dim=-1)
    margin_logit_mean = (top2_logits[:, 0] - top2_logits[:, 1]).mean()

    # Probabilities and Entropy
    log_probs = F.log_softmax(logits_2d, dim=-1)
    probs = log_probs.exp()

    # Full Entropy: H(p) = -sum(p * log_p)
    entropy_mean = -(probs * log_probs).sum(dim=-1).mean()

    # Top-1 Probability (confidence)
    top1_prob_mean = probs.max(dim=-1).values.mean()

    return {
        "loss_std": loss_std,
        "loss": loss_mean,
        "token_acc": token_acc,
        "entropy": entropy_mean,
        "top1_prob": top1_prob_mean,
        "margin_logit": margin_logit_mean,
    }


def process_sample(tokenizer, sample):
    query_messages = [
        {"role": "user", "content": sample["query"]},
    ]
    response_messages = query_messages + [
        {"role": "assistant", "content": sample["response"]},
    ]
    query_text = tokenizer.apply_chat_template(
        query_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    full_text = tokenizer.apply_chat_template(
        response_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    query_ids = tokenizer(query_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    response_ids = full_ids[len(query_ids) :]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
    return {
        "query": query_text,
        "response": response_text,
        "query_ids": query_ids,
        "response_ids": response_ids,
    }


def validate(args):
    if args.model:
        ckpt_path = args.model
        print(f"Using custom model path: {ckpt_path}")
    else:
        ckpt_path = f"ckpts/{args.run_id}"
        if not Path(ckpt_path).exists():
            print(f"Error: Checkpoint path '{ckpt_path}' does not exist.")
            print(f"Available checkpoints in ckpts/:")
            ckpts_dir = Path("ckpts")
            if ckpts_dir.exists():
                for item in sorted(ckpts_dir.iterdir()):
                    if item.is_dir():
                        print(f"  - {item.name}")
            return

    print(f"Loading model from {ckpt_path}...")

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    generation_config = GenerationConfig.from_pretrained(ckpt_path)

    model, _ = FastLanguageModel.from_pretrained(
        ckpt_path,
        load_in_4bit=False,
        full_finetuning=True,
        dtype=torch.bfloat16,
        max_seq_length=32768,
    )
    model = model.cuda()
    model.eval()
    model.generation_config = generation_config
    print(f"Model loaded successfully from {ckpt_path}")

    print(f"Loading validation set from {args.val_set}...")
    if Path(args.val_set).is_dir():
        val_set = load_from_disk(args.val_set)
    else:
        val_set = load_dataset(args.val_set, split="train")

    val_set = val_set.map(lambda x: process_sample(tokenizer, x), num_proc=64)
    print(f"Loaded validation set with {len(val_set)} samples")

    print("\n" + "=" * 80)
    print("First sample:")
    print("-" * 80)
    print("Query:")
    print(val_set[0]["query"])
    print("-" * 80)
    print("Response:")
    print(val_set[0]["response"])
    print("=" * 80)

    print("\nRunning validation...")
    all_metrics = []

    with torch.no_grad():
        for val_sample in tqdm(val_set, desc="Validating"):
            val_query_len = len(val_sample["query_ids"])
            val_input_ids = (
                torch.tensor(val_sample["query_ids"] + val_sample["response_ids"])
                .unsqueeze(0)
                .cuda()
            )
            val_logits = model(input_ids=val_input_ids).logits[
                :, val_query_len - 1 : -1
            ]
            val_token_metrics = compute_token_metrics(
                val_logits, val_input_ids[:, val_query_len:]
            )
            all_metrics.append(val_token_metrics)

    print("\n" + "=" * 80)
    print("VALIDATION METRICS")
    print("=" * 80)

    aggregated = {}
    for key in all_metrics[0].keys():
        values = [m[key].item() for m in all_metrics]
        mean_val = sum(values) / len(values)
        aggregated[f"{args.prefix}/{key}"] = mean_val
        print(f"{args.prefix}/{key}: {mean_val:.6f}")

    aggregated[f"{args.prefix}/source"] = "val.py"

    print("=" * 80)
    print(
        f"\nMain metric - {args.prefix}/loss: {aggregated[f'{args.prefix}/loss']:.6f}"
    )
    print("=" * 80)

    if args.update_wandb:
        import wandb

        print(f"\nUpdating wandb run with run_id: {args.run_id}...")
        api = wandb.Api()
        runs = api.runs(args.wandb_project, filters={"config.run_id": args.run_id})
        run = next(iter(runs), None)

        if run is None:
            print(f"Warning: No wandb run found with config.run_id == {args.run_id}")
        else:
            for k, v in aggregated.items():
                run.summary[k] = v
            run.summary.update()
            print(
                f"✓ Updated wandb run ({args.run_id}) with {len(aggregated)} metrics."
            )

    return aggregated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate a trained model on a validation set"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Run ID to load model from (loads from ckpts/{run_id})",
    )
    parser.add_argument(
        "--val_set",
        type=str,
        required=True,
        help="Path to validation dataset (local directory or HF dataset path)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Direct path to model (e.g., 'Qwen/Qwen3-8B-Base'). If not provided, uses ckpts/{run_id}",
    )

    parser.add_argument(
        "--update_wandb",
        action="store_true",
        help="Update wandb run with validation metrics",
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="data-repetition",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="val",
        help="Prefix for metric names (default: val)",
    )

    args = parser.parse_args()

    print("Validation Configuration:")
    print(f"  run_id: {args.run_id}")
    print(f"  val_set: {args.val_set}")
    print(f"  model: {args.model or f'ckpts/{args.run_id}'}")
    print(f"  update_wandb: {args.update_wandb}")
    print(f"  prefix: {args.prefix}")
    print()

    validate(args)
