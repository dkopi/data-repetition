import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse
from transformers import GenerationConfig, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from datasets import load_dataset, Dataset
from tqdm import tqdm
import regex as re
from functools import partial
from pathlib import Path


def extract_solution(sample):
    gt = sample["solution"]
    if "\\boxed{" not in gt:
        return {"query": sample["problem"], "solution": None}
    gt = gt.split("\\boxed{")[-1].split("}")[0]
    gt = gt.strip()
    return {"query": sample["problem"], "solution": gt}


def is_valid(sample):
    _INT_RE = re.compile(r"^[+-]?\d+$")
    _FRAC_RE = re.compile(r"^-?\\(?:d)?frac\s*\{\s*[+-]?\d+\s*\}\s*\{\s*[+-]?\d+\s*\}$")
    _CHAR_RE = re.compile(r"^[A-Za-z]$")

    gt = sample["solution"]
    if gt is None:
        return False

    if _INT_RE.fullmatch(gt):  # int
        return True
    if _FRAC_RE.fullmatch(gt):  # \frac{int}{int} or \dfrac{int}{int}
        return True
    if _CHAR_RE.fullmatch(gt):  # single letter
        return True
    return False


def get_dataset():
    ds = load_dataset("AI-MO/NuminaMath-TIR", split="train")
    ds = ds.map(extract_solution, num_proc=64)
    ds = ds.filter(is_valid, num_proc=64)
    return list(ds["query"]), list(ds["solution"])


def distill(args):
    queries, solutions = get_dataset()

    if args.samples is not None:
        queries = queries[: args.samples]
        solutions = solutions[: args.samples]

    tokenizer_path = args.model if args.tokenizer is None else args.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    generation_config = GenerationConfig.from_pretrained(tokenizer_path)
    if args.greedy:
        generation_config.temperature = 0.0
        generation_config.top_p = 1.0
        generation_config.top_k = -1
    eos_ids = (
        generation_config.eos_token_id
        if isinstance(generation_config.eos_token_id, list)
        else [generation_config.eos_token_id]
    )

    if args.gpus > 1:
        llm = LLM(
            model=args.model,
            tokenizer=tokenizer_path,
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_ratio,
            tensor_parallel_size=args.gpus,
            distributed_executor_backend="mp",
        )
    else:
        llm = LLM(
            model=args.model,
            tokenizer=tokenizer_path,
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_ratio,
        )

    prompts = []
    for q in queries:
        messages = [{"role": "user", "content": q}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt = TokensPrompt(prompt_token_ids=prompt)
        prompts.append(prompt)

    gen_out = llm.generate(
        prompts=prompts,
        sampling_params=SamplingParams(
            n=1,
            max_tokens=args.tokens,
            temperature=(
                generation_config.temperature
                if hasattr(generation_config, "temperature")
                else 0.0
            ),
            top_p=(
                generation_config.top_p if hasattr(generation_config, "top_p") else 1.0
            ),
            top_k=(
                generation_config.top_k if hasattr(generation_config, "top_k") else -1
            ),
            stop_token_ids=eos_ids,
        ),
        use_tqdm=partial(tqdm, smoothing=0.01),
    )

    _indices = []
    _queries = []
    _solutions = []
    _responses = []
    for i, prompt in enumerate(tqdm(gen_out)):
        for output in prompt.outputs:
            terminated = output.token_ids[-1] in eos_ids
            text = tokenizer.decode(
                output.token_ids,
                skip_special_tokens=False,
            )
            if terminated:
                _indices.append(i)
                _queries.append(queries[i])
                _solutions.append(solutions[i])
                _responses.append(text)

    ds = Dataset.from_dict(
        {
            "idx": _indices,
            "query": _queries,
            "response": _responses,
            "solution": _solutions,
        }
    )
    print("===== Dataset Sample =====")
    print("Query:")
    print(ds[0]["query"])
    print("\nResponse:")
    print(ds[0]["response"])
    print("\nSolution:")
    print(ds[0]["solution"])
    ds.save_to_disk(args.out)
    print(f"Saved distilled dataset to {Path(args.out).absolute()}...")
    print(f"Total samples: {len(ds)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--gpu_ratio", type=float, default=0.9)
    parser.add_argument("--tokens", type=int, default=10000)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--gpus", type=int, default=1)

    args = parser.parse_args()

    distill(args)
