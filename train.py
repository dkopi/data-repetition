import unsloth
from unsloth import FastLanguageModel
import argparse
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    get_cosine_schedule_with_warmup,
)
import torch
import torch.nn.functional as F
from datasets import load_from_disk, load_dataset
import time
import bitsandbytes as bnb
from collections import deque
from tqdm import tqdm
import os
import wandb
from pathlib import Path
from peft import LoraConfig, get_peft_model


def compute_token_metrics(
    logits: torch.Tensor,  # [B, T, V]
    targets: torch.Tensor,  # [B, T]
):
    # Flatten to [N, V] and [N] where N = B*T
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


def train(args):
    if args.wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            group=args.wandb_group,
            config=vars(args),
        )

    tokenizer_path = args.model if args.tokenizer is None else args.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    generation_config = GenerationConfig.from_pretrained(tokenizer_path)

    model, _ = FastLanguageModel.from_pretrained(
        args.model,
        load_in_4bit=False,
        full_finetuning=not args.lora,
        dtype=torch.bfloat16,
        max_seq_length=32768,
    )

    if args.lora:
        _layers = [
            n
            for n, p in model.named_modules()
            if isinstance(p, torch.nn.Linear) and "layers" in n
        ]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=(args.lora_r if args.lora_alpha is None else args.lora_alpha),
            lora_dropout=0.0,
            target_modules=_layers,
        )
        model = get_peft_model(model, lora_config)
        model = model.bfloat16()

    model = model.cuda()
    model.train()
    model.generation_config = generation_config
    print(f"Loaded model {args.model} for training...")

    if not args.lora:
        for p in model.parameters():
            p.requires_grad = True
    print(
        f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    if Path(args.dataset).is_dir():
        dataset = load_from_disk(args.dataset)
    else:
        dataset = load_dataset(args.dataset, split="train")
    dataset = dataset.map(lambda x: process_sample(tokenizer, x), num_proc=64)
    print(f"Loaded dataset with {len(dataset)} samples from {args.dataset}...")

    val_set = None
    if args.val_set is not None:
        if Path(args.val_set).is_dir():
            val_set = load_from_disk(args.val_set)
        else:
            val_set = load_dataset(args.val_set, split="train")
        val_set = val_set.map(lambda x: process_sample(tokenizer, x), num_proc=64)
        print(
            f"Loaded validation set with {len(val_set)} samples from {args.val_set}..."
        )

    optimizer = bnb.optim.PagedAdamW8bit(
        model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, args.beta2)
    )

    steps_per_epoch = len(dataset)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    rolling_loss = deque(maxlen=100)
    total_tokens = 0
    total_response_tokens = 0
    metrics = None
    total_steps = 0

    for epoch in range(args.epochs):
        dataset = dataset.shuffle(seed=args.seed + epoch)
        progress_bar = tqdm(range(steps_per_epoch), smoothing=0.01)
        for step_i in progress_bar:
            sample = dataset[step_i]
            query_len = len(sample["query_ids"])
            response_len = len(sample["response_ids"])
            total_tokens += query_len + response_len
            total_response_tokens += response_len

            if epoch == 0 and step_i == 0:
                print("Query:")
                print(sample["query"])
                print("Response:")
                print(sample["response"])

            input_ids = (
                torch.tensor(sample["query_ids"] + sample["response_ids"])
                .unsqueeze(0)
                .cuda()
            )
            logits = model(input_ids=input_ids).logits[:, query_len - 1 : -1]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                input_ids[:, query_len:].reshape(-1),
            )
            if args.acc_steps is not None:
                (loss / args.acc_steps).backward()
            else:
                loss.backward()

            l2_norm = None
            if args.acc_steps is None or (step_i + 1) % args.acc_steps == 0:
                l2_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()

            _loss = loss.detach().cpu().item()
            rolling_loss.append(_loss)
            _rolling_loss = sum(rolling_loss) / len(rolling_loss)
            progress_bar.set_description(
                f"epoch: {epoch} step: {step_i} loss: {_rolling_loss:.4f} lr: {scheduler.get_last_lr()[0]:.2e}"
            )

            if metrics is None:
                metrics = {
                    "loss": [],
                    "grad_l2_norm": [],
                }
            if l2_norm is not None:
                metrics["grad_l2_norm"].append(l2_norm.detach().cpu().item())
            metrics["loss"].append(_loss)
            with torch.no_grad():
                token_metrics = compute_token_metrics(logits, input_ids[:, query_len:])
            for k, v in token_metrics.items():
                _key = f"train/{k}"
                if _key not in metrics:
                    metrics[_key] = []
                metrics[_key].append(v.detach().cpu().item())

            if val_set is not None and (
                step_i == steps_per_epoch - 1 and epoch == args.epochs - 1
            ):
                with torch.no_grad():
                    sums = None
                    n = 0

                    for val_sample in tqdm(val_set):
                        val_query_len = len(val_sample["query_ids"])
                        val_input_ids = (
                            torch.tensor(
                                val_sample["query_ids"] + val_sample["response_ids"]
                            )
                            .unsqueeze(0)
                            .cuda()
                        )
                        val_logits = model(input_ids=val_input_ids).logits[
                            :, val_query_len - 1 : -1
                        ]
                        m = compute_token_metrics(
                            val_logits, val_input_ids[:, val_query_len:]
                        )

                        if sums is None:
                            sums = {kk: 0.0 for kk in m.keys()}

                        for kk, vv in m.items():
                            sums[kk] += float(vv.detach().cpu())
                        n += 1

                    val_token_metrics = {kk: sums[kk] / n for kk in sums}

                for k, v in val_token_metrics.items():
                    _key = f"val/{k}"
                    if _key not in metrics:
                        metrics[_key] = []
                    metrics[_key].append(v)

            if (
                (step_i + 1) % args.wandb_freq == 0
                or step_i == 0
                or step_i == steps_per_epoch - 1
                and epoch == args.epochs - 1
            ):
                avg_metrics = {
                    k: sum(v) / len(v) for k, v in metrics.items() if len(v) > 0
                }
                avg_metrics["train/epoch"] = epoch
                avg_metrics["train/step"] = step_i + epoch * steps_per_epoch
                avg_metrics["train/total_tokens"] = total_tokens
                avg_metrics["train/total_response_tokens"] = total_response_tokens
                for k, v in avg_metrics.items():
                    print(f"{k}: {v}")
                if args.wandb:
                    wandb.log(avg_metrics)
                metrics = None

            total_steps += 1

    if not args.skip_saving:
        os.makedirs("ckpts", exist_ok=True)
        model.save_pretrained(f"ckpts/{args.run_id}")
        tokenizer.save_pretrained(f"ckpts/{args.run_id}")
        print(f"Saved model checkpoint to ckpts/{args.run_id}")

    if args.wandb:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--val_set", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_saving", action="store_true")
    parser.add_argument("--acc_steps", type=int, default=None)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="data-repetition")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_freq", type=int, default=100)

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beta2", type=float, default=0.999)

    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=None)

    args = parser.parse_args()

    if args.run_id is None:
        args.run_id = str(int(time.time()))

    print("args:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    train(args)
