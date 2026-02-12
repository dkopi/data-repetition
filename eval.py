import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse
from transformers import GenerationConfig, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest
from datasets import load_dataset
import re
from tqdm import tqdm
from peft import PeftConfig


def get_aime24():
    ds = load_dataset("math-ai/aime24", split="test")
    ds = ds.map(
        lambda x: {
            "query": x["problem"]
            + "\n\nAnswer the query and place the final integer inside \\boxed{} in your response. For example \\boxed{42}.",
            "solution": x["solution"].split("\\boxed{")[-1].split("}")[0],
        }
    )
    return list(ds["query"]), list(ds["solution"])


def get_aime25():
    ds = load_dataset("math-ai/aime25", split="test")
    ds = ds.map(
        lambda x: {
            "query": x["problem"]
            + "\n\nAnswer the query and place the final integer inside \\boxed{} in your response. For example \\boxed{42}.",
            "solution": x["answer"].strip(),
        }
    )
    return list(ds["query"]), list(ds["solution"])


def get_gpqa():
    ds = load_dataset("dakopi/gpqa_diamond", split="train")
    ds = ds.map(
        lambda x: {
            "query": x["query"]
            + "\n\nAnswer the query and place the letter of chosen solution inside \\boxed{} in your response. For example \\boxed{A}.",
            "solution": x["solution"],
        }
    )
    return list(ds["query"]), list(ds["solution"])


def evaluate_model(args):

    if args.task == "aime24":
        queries, solutions = get_aime24()
    elif args.task == "aime25":
        queries, solutions = get_aime25()
    elif args.task == "gpqa":
        queries, solutions = get_gpqa()
    else:
        raise NotImplementedError(f"Task {args.task} not implemented.")

    if args.samples is not None:
        queries = queries[: args.samples]
        solutions = solutions[: args.samples]

    tokenizer_path = args.model if args.tokenizer is None else args.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if os.path.exists(os.path.join(tokenizer_path, "generation_config.json")):
        generation_config = GenerationConfig.from_pretrained(tokenizer_path)
    else:
        print("generation_config.json not found, using default generation config.")
        generation_config = GenerationConfig()
        generation_config.eos_token_id = tokenizer.eos_token_id
        generation_config.pad_token_id = tokenizer.pad_token_id
        generation_config.bos_token_id = tokenizer.bos_token_id
        generation_config.temperature = 0.6
        generation_config.top_p = 0.95
        generation_config.top_k = -1

    if args.greedy:
        generation_config.temperature = 0.0
        generation_config.top_p = 1.0
        generation_config.top_k = -1
    eos_ids = (
        generation_config.eos_token_id
        if isinstance(generation_config.eos_token_id, list)
        else [generation_config.eos_token_id]
    )

    peft_config = None
    if os.path.exists(os.path.join(args.model, "adapter_config.json")):
        peft_config = PeftConfig.from_pretrained(args.model)

    if peft_config is not None:
        llm = LLM(
            model=peft_config.base_model_name_or_path,
            tokenizer=tokenizer_path,
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_ratio,
            enable_lora=True,
            max_lora_rank=max(peft_config.r, 8),
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
            n=args.n,
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
        use_tqdm=True,
        lora_request=(
            LoRARequest("adapter", 1, args.model) if peft_config is not None else None
        ),
    )

    n = args.n
    metrics = {
        "length": [],
        f"avg@{n}": [],
        "terminated": [],
    }
    for _n in range(n):
        metrics[f"pass@{_n+1}"] = []

    boxed = re.compile(r"\\boxed\s*\{([^{}]*)\}")
    for i, prompt in enumerate(tqdm(gen_out)):
        _correct = []
        for output in prompt.outputs:
            metrics["length"].append(float(len(output.token_ids)))
            metrics["terminated"].append(float(output.token_ids[-1] in eos_ids) * 100.0)
            text = tokenizer.decode(
                output.token_ids,
                skip_special_tokens=False,
            )
            try:
                content = boxed.findall(text)[-1]
            except:
                content = ""
            _correct.append(content == solutions[i])
        metrics[f"avg@{n}"].append(sum(_correct) / len(_correct) * 100.0)
        for _n in range(1, n + 1):
            metrics[f"pass@{_n}"].append(any(_correct[:_n]) * 100.0)

    metrics[f"avg@{args.n}"] = sum(metrics[f"avg@{args.n}"]) / len(
        metrics[f"avg@{args.n}"]
    )
    for _n in range(1, n + 1):
        metrics[f"pass@{_n}"] = sum(metrics[f"pass@{_n}"]) / len(metrics[f"pass@{_n}"])
    metrics["length"] = sum(metrics["length"]) / len(metrics["length"])
    metrics["terminated"] = sum(metrics["terminated"]) / len(metrics["terminated"])

    if args.print:
        for i, prompt in enumerate(gen_out):
            print(f"=== Sample {i} ===")
            print("query:")
            print(
                tokenizer.decode(
                    prompts[i]["prompt_token_ids"], skip_special_tokens=False
                )
            )
            print("response:")
            print(
                tokenizer.decode(prompt.outputs[0].token_ids, skip_special_tokens=False)
            )
            print("terminated:", prompt.outputs[0].token_ids[-1] in eos_ids)

    else:
        print("first query:")
        print(
            tokenizer.decode(prompts[0]["prompt_token_ids"], skip_special_tokens=False)
        )
        print("first response:")
        print(
            tokenizer.decode(gen_out[0].outputs[0].token_ids, skip_special_tokens=False)
        )

    print(f"Evaluation results for model {args.model} on task {args.task}:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if args.run_id is not None:
        import wandb

        api = wandb.Api()
        runs = api.runs(args.wandb_project, filters={"config.run_id": args.run_id})
        run = next(iter(runs), None)

        if run is None:
            print(f"No run found with run_id == {args.run_id}")
        else:
            for k, v in metrics.items():
                metric_key = f"{args.task}/{k}"
                run.summary[metric_key] = v
            run.summary[f"{args.task}/samples"] = len(queries)
            run.summary[f"{args.task}/n"] = args.n
            run.summary.update()
            print(f"Updated wandb run ({args.run_id}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--task", type=str)
    parser.add_argument("--gpu_ratio", type=float, default=0.9)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=30000)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--print", action="store_true")

    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="data-repetition")

    args = parser.parse_args()

    print("args:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    evaluate_model(args)
