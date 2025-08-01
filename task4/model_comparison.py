import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import ollama

# === Configuration ===
friendly_prompt_file = "./friendly.txt"
test_cases_file = "./test_cases.json"
finetuned_model_path = "./phi2-finetuned"
base_model_name = "microsoft/phi-2"
original_ollama_model = "phi"  # Original Phi model in Ollama

class ModelComparator:
    def __init__(self):
        self.finetuned_model = None
        self.finetuned_tokenizer = None
        self.load_finetuned_model()
    
    def load_finetuned_model(self):
        """Load the fine-tuned Phi-2 model with LoRA weights"""
        try:
            if not os.path.exists(finetuned_model_path):
                print(f"❌ Fine-tuned model not found at {finetuned_model_path}")
                print("Run finetune.py first to create the fine-tuned model")
                return
            
            print("Loading fine-tuned model...")
            
            # Load tokenizer
            self.finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
            if self.finetuned_tokenizer.pad_token is None:
                self.finetuned_tokenizer.pad_token = self.finetuned_tokenizer.eos_token
            
            # Load base model (CPU only to avoid MPS issues)
            print("Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                device_map=None,  # No device mapping
                low_cpu_mem_usage=True
            )
            
            # Force model to CPU
            base_model = base_model.to("cpu")
            
            # Load LoRA weights
            print("Loading LoRA weights...")
            self.finetuned_model = PeftModel.from_pretrained(base_model, finetuned_model_path)
            
            # Ensure final model is on CPU
            self.finetuned_model = self.finetuned_model.to("cpu")
            self.finetuned_model.eval()
            
            print("✅ Fine-tuned model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading fine-tuned model: {e}")
            self.finetuned_model = None
    
    def generate_original_response(self, prompt: str) -> str:
        """Generate response using original Phi model via Ollama"""
        try:
            response = ollama.chat(
                model=original_ollama_model, 
                messages=[{"role": "user", "content": prompt}]
            )
            return response['message']['content'].strip()
        except Exception as e:
            return f"Error with original model: {str(e)}"
    
    def generate_finetuned_response(self, text, friendly_prompt):
        """Generate response using fine-tuned model"""
        try:
            # Ensure CPU-only execution
            prompt = f"{friendly_prompt}\n\nCustomer complaint: {text}\n\nResponse:"
            
            inputs = self.finetuned_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            # Explicitly place on CPU
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.finetuned_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.finetuned_tokenizer.eos_token_id
                )
            
            # Decode only the new tokens
            response = self.finetuned_tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            return f"Error with fine-tuned model: {str(e)}"

def load_friendly_prompt():
    """Load the friendly prompt template"""
    with open(friendly_prompt_file, 'r', encoding='utf-8') as f:
        return f.read().strip()

def load_test_complaints():
    """Load test complaints from JSON file"""
    with open(test_cases_file, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    # Extract just the complaints
    complaints = [test_case['vars']['complaint'] for test_case in test_cases]
    return complaints

def compare_models_on_complaint(complaint: str, friendly_prompt: str, comparator: ModelComparator):
    """Compare both models on a single complaint"""
    
    # Create the full prompt by replacing the placeholder
    full_prompt = friendly_prompt.replace("{{complaint}}", complaint)
    
    print(f"  Generating responses...")
    
    # Generate responses from both models
    original_response = comparator.generate_original_response(full_prompt)
    finetuned_response = comparator.generate_finetuned_response(complaint, friendly_prompt)
    
    return {
        "complaint": complaint,
        "prompt_used": full_prompt,
        "original_model": {
            "name": original_ollama_model,
            "response": original_response
        },
        "finetuned_model": {
            "name": f"Fine-tuned {base_model_name}",
            "response": finetuned_response
        }
    }

def main():
    print("=== Phi-2 Model Comparison Framework ===")
    print("Comparing original Phi vs fine-tuned Phi-2 using friendly prompt")
    
    # Initialize model comparator
    print("\n1. Loading models...")
    comparator = ModelComparator()
    
    # Load prompt and test cases
    print("\n2. Loading prompt and test cases...")
    friendly_prompt = load_friendly_prompt()
    test_complaints = load_test_complaints()
    
    print(f"✅ Loaded friendly prompt from {friendly_prompt_file}")
    print(f"✅ Loaded {len(test_complaints)} test complaints")
    
    # Run comparisons
    print(f"\n3. Running model comparisons...")
    print("=" * 60)
    
    results = []
    
    for i, complaint in enumerate(test_complaints, 1):
        print(f"\n[{i}/{len(test_complaints)}] Processing: {complaint}")
        
        result = compare_models_on_complaint(complaint, friendly_prompt, comparator)
        results.append(result)
        
        print(f"  Original: {result['original_model']['response'][:80]}...")
        print(f"  Fine-tuned: {result['finetuned_model']['response'][:80]}...")
    
    # Save results
    output_file = "model_comparison_results.json"
    print(f"\n4. Saving results...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"COMPARISON COMPLETE!")
    print(f"Total complaints tested: {len(results)}")
    print(f"Results saved to: {output_file}")
    print(f"\nModel comparison:")
    print(f"  Original: {original_ollama_model} (via Ollama)")
    print(f"  Fine-tuned: {base_model_name} + LoRA weights")
    print(f"\nNext steps:")
    print(f"1. Review {output_file} for detailed comparison")
    print(f"2. Analyze response quality differences")
    print(f"3. Look for improvements from fine-tuning on Alpaca/HH data")

if __name__ == "__main__":
    main()
