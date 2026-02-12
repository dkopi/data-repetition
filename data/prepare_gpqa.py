import os
from datasets import load_dataset
import random

random.seed(42)

ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")


def parse_sample(sample):
    query = sample["Question"].strip()
    letters = ["A", "B", "C", "D"]
    correct_answer = sample["Correct Answer"]
    choices = [
        correct_answer,
        sample["Incorrect Answer 1"],
        sample["Incorrect Answer 2"],
        sample["Incorrect Answer 3"],
    ]
    random.shuffle(choices)
    correct_idx = choices.index(correct_answer)
    solution = letters[correct_idx]
    query += "\n\nChoices:"
    for i, choice in enumerate(choices):
        query += f"\n{letters[i]}: " + choice.strip()

    return {"query": query, "solution": solution}


ds = ds.map(parse_sample)
ds = ds.remove_columns(
    [col for col in ds.column_names if col not in ["query", "solution"]]
)

ds.push_to_hub("gpqa_diamond")
