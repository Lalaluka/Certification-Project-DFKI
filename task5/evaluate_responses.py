import json
import os
from bert_score import score

# Path to responses
RESPONSES_DIR = "responses"
BASELINE_VARIANT = "prompt_only"
COMPARE_VARIANTS = ["rag", "fine_tuned", "agent"]

# Dictionary to store results
results = {}

# Load prompt_only as reference
with open(os.path.join(RESPONSES_DIR, f"{BASELINE_VARIANT}.json"), "r") as f:
    baseline_data = json.load(f)
    references = [entry["response"] for entry in baseline_data]

for variant in COMPARE_VARIANTS:
    with open(os.path.join(RESPONSES_DIR, f"{variant}.json"), "r") as f:
        variant_data = json.load(f)
        candidates = [entry["response"] for entry in variant_data]

    print(f"\n🔍 Comparing {variant} vs {BASELINE_VARIANT} (BERTScore):")

    P, R, F1 = score(candidates, references, lang="en", model_type="roberta-large")
    
    # Store variant results
    variant_results = {
        "compared_to": BASELINE_VARIANT,
        "cases": [],
        "avg_f1": float(F1.mean())
    }
    
    for i, (p, r, f) in enumerate(zip(P, R, F1)):
        print(f"  [Case {i+1}] P={p:.3f}, R={r:.3f}, F1={f:.3f}")
        variant_results["cases"].append({
            "case": i+1,
            "precision": float(p),
            "recall": float(r),
            "f1": float(f)
        })

    print(f"  ▶️  Avg F1: {F1.mean():.3f}")
    results[variant] = variant_results

# Save results to JSON file
output_path = os.path.join("", "bertscore_results.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_path}")
