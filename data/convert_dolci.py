from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct")

ds = load_dataset("allenai/Dolci-Think-SFT-7B", split="train")
ds = ds.shuffle(seed=42)
ds = ds.select(range(200000))
ds = ds.filter(
    lambda x: x["messages"][0]["role"] == "user"
    and x["messages"][1]["role"] == "assistant",
    num_proc=64,
)
ds = ds.map(
    lambda x: {
        "query": x["messages"][0]["content"],
        "response": x["messages"][1]["content"],
        "solution": None,
    },
    num_proc=64,
)
ds = ds.remove_columns(
    [col for col in ds.column_names if col not in ["query", "response", "solution"]]
)
ds = ds.filter(
    lambda x: "<think>" in x["response"] and "</think>" in x["response"], num_proc=64
)
ds = ds.filter(
    lambda x: len(tokenizer(x["response"])["input_ids"]) < 10000, num_proc=64
)

ds.save_to_disk("datasets/dolci_think")
