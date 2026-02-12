import argparse
from datasets import load_from_disk


def filter(sample):
    return sample["solution"].strip().isdigit()


def process(sample):
    boxed = sample["response"].split("\\boxed{")[-1].split("}")[0].strip()
    correct = boxed == sample["solution"].strip()
    return {
        "correct": correct,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Shorten dataset responses by removing thinking tokens and end tokens."
    )
    parser.add_argument(
        "input_path", type=str, help="Path to the input dataset directory"
    )
    parser.add_argument(
        "output_path", type=str, help="Path to save the processed dataset"
    )

    args = parser.parse_args()

    print(f"Loading dataset from {args.input_path}...")
    ds = load_from_disk(args.input_path)

    print("Filtering dataset...")
    ds = ds.filter(filter)

    print("Processing responses...")
    ds = ds.map(process)

    positive = ds.filter(lambda x: x["correct"])
    negative = ds.filter(lambda x: not x["correct"])

    print(f"Saving processed dataset to {args.output_path}...")
    positive.save_to_disk(args.output_path + "__positive")
    negative.save_to_disk(args.output_path + "__negative")

    print("Done!")


if __name__ == "__main__":
    main()
