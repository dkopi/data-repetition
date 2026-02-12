from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
from unsloth import FastLanguageModel
import time

MMLU_DOMAIN_TO_SUBJECTS = {
    "stem": [
        "abstract_algebra",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    ],
    "humanities": [
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "professional_law",
        "world_religions",
    ],
    "social_sciences": [
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
    "other": [
        "anatomy",
        "business_ethics",
        "clinical_knowledge",
        "college_medicine",
        "global_facts",
        "human_aging",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_accounting",
        "professional_medicine",
        "virology",
    ],
}


def evaluate_subject(subject, model, tokenizer, device, args):
    for attempt in range(3):
        try:
            full_dataset = load_dataset("cais/mmlu", subject)
            break
        except Exception as e:
            if attempt < 2:
                print(f"Error loading dataset for subject '{subject}': {e}")
                print("Retrying in 5 minutes...")
                time.sleep(300)
            else:
                raise e

    def format(sample, few_shot=None):
        def template(sample):
            query = f"{sample['question']}\nAnswer with the letter enclosed in \\boxed{{}}.\n\nChoices:\nA: {sample['choices'][0]}\nB: {sample['choices'][1]}\nC: {sample['choices'][2]}\nD: {sample['choices'][3]}"
            return query

        messages = []
        if few_shot is not None:
            for fs in few_shot:
                messages.append(
                    {
                        "role": "user",
                        "content": template(fs),
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": "\\boxed{"
                        + ["A", "B", "C", "D"][fs["answer"]]
                        + "}",
                    }
                )
        messages.append(
            {
                "role": "user",
                "content": template(sample),
            }
        )
        query = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        query = query + "\\boxed{"
        return {"query": query}

    dataset = full_dataset["test"].map(
        lambda x: format(
            x,
            few_shot=(
                full_dataset["dev"].select(range(args.few_shot))
                if args.few_shot > 0
                else None
            ),
        )
    )
    dataset = dataset.map(lambda x: tokenizer(x["query"], add_special_tokens=False))

    a_idx = tokenizer.encode("A")[0]
    b_idx = tokenizer.encode("B")[0]
    c_idx = tokenizer.encode("C")[0]
    d_idx = tokenizer.encode("D")[0]

    results = []

    with torch.no_grad():
        progress_bar = tqdm(dataset)
        for i, sample in enumerate(progress_bar):
            if i == 0:
                print(sample["query"])

            input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(device)
            logits = model(input_ids).logits
            probs = torch.softmax(logits[0, -1], dim=-1)
            top1 = probs.argmax()
            top1_tok = tokenizer.decode([top1])
            a = probs[a_idx]
            b = probs[b_idx]
            c = probs[c_idx]
            d = probs[d_idx]
            solution = sample["answer"]

            if args.print:
                print(f"top1 tok: {top1_tok}")
                print(f"A: {int(a*100)}")
                print(f"B: {int(b*100)}")
                print(f"C: {int(c*100)}")
                print(f"D: {int(d*100)}")
                print(f"-- solution: {['A','B','C','D'][solution]}\n")

            if solution == 0:
                correct = a == max([a, b, c, d])
            elif solution == 1:
                correct = b == max([a, b, c, d])
            elif solution == 2:
                correct = c == max([a, b, c, d])
            elif solution == 3:
                correct = d == max([a, b, c, d])
            else:
                raise NotImplementedError()

            results.append(correct)

    accuracy = sum(results) / len(results) * 100 if len(results) > 0 else 0.0
    return accuracy


def run(args):
    device = "cuda"

    if args.run_id:
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
        model, _ = FastLanguageModel.from_pretrained(
            ckpt_path,
            load_in_4bit=False,
            full_finetuning=True,
            dtype=torch.bfloat16,
            max_seq_length=32768,
        )
        model = model.cuda()
        model.eval()
        print(f"Model loaded successfully from {ckpt_path}")
    else:
        model_path = args.model
        tokenizer_path = args.tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    for p in model.parameters():
        p.requires_grad = False

    subject_accuracies = {}
    domain_accuracies = {}

    for domain, subjects in MMLU_DOMAIN_TO_SUBJECTS.items():
        print(f"\n{'='*80}")
        print(f"Evaluating domain: {domain.upper()}")
        print(f"{'='*80}")

        domain_results = []

        for subject in subjects:
            print(f"\nEvaluating subject: {subject}")
            accuracy = evaluate_subject(subject, model, tokenizer, device, args)
            subject_accuracies[subject] = accuracy
            domain_results.append(accuracy)
            print(f"  {subject}: {accuracy:.2f}%")

        domain_acc = (
            sum(domain_results) / len(domain_results)
            if len(domain_results) > 0
            else 0.0
        )
        domain_accuracies[domain] = domain_acc

    overall_acc = (
        sum(subject_accuracies.values()) / len(subject_accuracies)
        if len(subject_accuracies) > 0
        else 0.0
    )

    print(f"\n\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")

    print("\n--- Per Subject Accuracies ---")
    for domain, subjects in MMLU_DOMAIN_TO_SUBJECTS.items():
        print(f"\n{domain.upper()}:")
        for subject in subjects:
            print(f"  {subject}: {subject_accuracies[subject]:.2f}%")

    print("\n--- Per Domain Accuracies ---")
    for domain, acc in domain_accuracies.items():
        print(f"  {domain}: {acc:.2f}%")

    print(f"\n--- Overall MMLU Accuracy ---")
    print(f"  Average: {overall_acc:.2f}%")
    print(f"{'='*80}\n")

    metrics = {
        "mmlu/acc": overall_acc,
    }
    for domain, acc in domain_accuracies.items():
        metrics[f"mmlu/{domain}/acc"] = acc

    if args.update_wandb:
        import wandb

        print(f"\nUpdating wandb run with run_id: {args.run_id}...")
        api = wandb.Api()
        runs = api.runs(args.wandb_project, filters={"config.run_id": args.run_id})
        run = next(iter(runs), None)

        if run is None:
            print(f"Warning: No wandb run found with config.run_id == {args.run_id}")
        else:
            for k, v in metrics.items():
                run.summary[k] = v
            run.summary.update()
            print(f"✓ Updated wandb run ({args.run_id}) with {len(metrics)} metrics.")

    return metrics


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run ID to load model from checkpoint (loads from ckpts/{run_id})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model path (used if --run_id is not provided)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HuggingFace tokenizer path (used if --run_id is not provided)",
    )
    parser.add_argument("--print", action="store_true")
    parser.add_argument("--few_shot", type=int, default=5)
    parser.add_argument(
        "--update_wandb",
        action="store_true",
        help="Update wandb run with MMLU metrics",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="data-repetition",
    )
    args = parser.parse_args()

    if not args.run_id and (not args.model or not args.tokenizer):
        parser.error("Either --run_id or both --model and --tokenizer must be provided")

    print("===========================")
    print("args:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("===========================")
    run(args)
