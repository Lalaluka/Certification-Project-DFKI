#!/usr/bin/env python3

import json
import torch
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# === CONFIG ===
BASE_MODEL_NAME = "microsoft/phi-2"
FINETUNED_MODEL_PATH = "../task4/experiments/model_1"
TEST_CASES_FILE = "../task1/test_cases.json"
FRIENDLY_PROMPT_FILE = "../task1/prompts/friendly.txt"

def calculate_perplexity(model, tokenizer, text: str) -> float:
    """Calculate perplexity for a given text"""
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    input_ids = inputs.input_ids
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    
    return perplexity

def load_test_scenarios():
    """Load customer service scenarios for testing"""
    with open(FRIENDLY_PROMPT_FILE, 'r') as f:
        prompt_template = f.read().strip()
    
    with open(TEST_CASES_FILE, 'r') as f:
        test_cases = json.load(f)
    
    scenarios = []
    for case in test_cases:
        if 'vars' in case and 'complaint' in case['vars']:
            complaint = case['vars']['complaint']
            prompt = prompt_template.replace("{{complaint}}", complaint)
            
            # Create realistic customer service response for testing
            response = f"I understand your concern about {complaint.lower()}. Let me help you resolve this issue quickly and effectively."
            full_scenario = f"{prompt}\n\n{response}"
            
            scenarios.append({
                "complaint": complaint,
                "text": full_scenario
            })
    
    return scenarios

def compare_models():
    """Compare base model vs fine-tuned model perplexity"""
    print("🔍 Quick Perplexity Comparison: Base vs Fine-tuned")
    print("=" * 55)
    
    # Load test scenarios
    scenarios = load_test_scenarios()
    print(f"📝 Loaded {len(scenarios)} test scenarios")
    
    # Load base model
    print("\n📥 Loading base model...")
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    base_model.eval()
    print("✅ Base model loaded")
    
    # Load fine-tuned model
    print(f"\n📥 Loading fine-tuned model from {FINETUNED_MODEL_PATH}...")
    finetuned_tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
    if finetuned_tokenizer.pad_token is None:
        finetuned_tokenizer.pad_token = finetuned_tokenizer.eos_token
    
    base_for_peft = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    finetuned_model = PeftModel.from_pretrained(base_for_peft, FINETUNED_MODEL_PATH)
    finetuned_model.eval()
    print("✅ Fine-tuned model loaded")
    
    # Calculate perplexities
    print(f"\n🧮 Calculating perplexities for {len(scenarios)} scenarios...")
    
    base_perplexities = []
    finetuned_perplexities = []
    
    for i, scenario in enumerate(scenarios):
        print(f"  [{i+1}/{len(scenarios)}] {scenario['complaint'][:40]}...")
        
        try:
            # Base model perplexity
            base_ppl = calculate_perplexity(base_model, base_tokenizer, scenario['text'])
            base_perplexities.append(base_ppl)
            
            # Fine-tuned model perplexity
            ft_ppl = calculate_perplexity(finetuned_model, finetuned_tokenizer, scenario['text'])
            finetuned_perplexities.append(ft_ppl)
            
        except Exception as e:
            print(f"    ⚠️ Error: {e}")
            continue
    
    # Calculate statistics
    base_avg = np.mean(base_perplexities)
    base_std = np.std(base_perplexities)
    
    ft_avg = np.mean(finetuned_perplexities)
    ft_std = np.std(finetuned_perplexities)
    
    improvement = ((base_avg - ft_avg) / base_avg) * 100
    
    # Results
    print("\n" + "=" * 55)
    print("📊 PERPLEXITY COMPARISON RESULTS")
    print("=" * 55)
    print(f"Base Model (microsoft/phi-2):")
    print(f"  Average Perplexity: {base_avg:.2f} (±{base_std:.2f})")
    print(f"  Range: {min(base_perplexities):.2f} - {max(base_perplexities):.2f}")
    
    print(f"\nFine-tuned Model:")
    print(f"  Average Perplexity: {ft_avg:.2f} (±{ft_std:.2f})")
    print(f"  Range: {min(finetuned_perplexities):.2f} - {max(finetuned_perplexities):.2f}")
    
    print(f"\n🎯 Performance Improvement:")
    if improvement > 0:
        print(f"  Fine-tuned model is {improvement:.1f}% better")
        print(f"  (Lower perplexity = better language modeling)")
    else:
        print(f"  Base model is {abs(improvement):.1f}% better")
        print(f"  (Fine-tuning may need adjustment)")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "base_model": {
            "name": BASE_MODEL_NAME,
            "avg_perplexity": base_avg,
            "std_perplexity": base_std,
            "min_perplexity": min(base_perplexities),
            "max_perplexity": max(base_perplexities)
        },
        "finetuned_model": {
            "path": FINETUNED_MODEL_PATH,
            "avg_perplexity": ft_avg,
            "std_perplexity": ft_std,
            "min_perplexity": min(finetuned_perplexities),
            "max_perplexity": max(finetuned_perplexities)
        },
        "improvement_percentage": improvement,
        "num_scenarios": len(scenarios)
    }
    
    with open("quick_perplexity_comparison.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: quick_perplexity_comparison.json")

if __name__ == "__main__":
    compare_models()
