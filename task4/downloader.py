from datasets import load_dataset

ds = load_dataset("tatsu-lab/alpaca")
ds["train"].shuffle(seed=42).select(range(2000)).to_json("data/alpaca_subset.json")

hh = load_dataset("Anthropic/hh-rlhf", split="train")
hh.to_json("data/hh_subset.json")
