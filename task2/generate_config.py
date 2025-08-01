import os
import json
import sys
import yaml

# Add shared utilities to path
sys.path.append('../shared')
from rag_utils import create_rag

# === CONFIG ===
promptfoo_output_path = "promptfooconfig.yaml"
top_k = 2
max_context_length = 300  # Maximum characters per retrieved document
prompts_dir = "../task1/prompts"
test_cases_file = "../task1/test_cases.json"

# Initialize unified RAG system
rag = create_rag()

def load_prompts_from_directory(prompts_dir):
    """Load prompts from the prompts directory"""
    prompts = []
    
    for filename in os.listdir(prompts_dir):
        if filename.endswith('.txt'):
            prompt_name = filename[:-4] 
            filepath = os.path.join(prompts_dir, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                prompts.append({
                    "id": f"{prompt_name}-no-rag",
                    "raw": content
                })
                
                rag_content = content.replace(
                    "Customer Complaint: {{complaint}}",
                    "Relevant information from our knowledge base:\n{{context}}\n\nCustomer Complaint: {{complaint}}"
                )
                
                prompts.append({
                    "id": f"{prompt_name}-rag",
                    "raw": rag_content
                })
    
    return prompts

def load_test_cases_from_file(filename):
    """Load and enhance test cases with RAG context using unified RAG"""
    with open(filename, "r", encoding="utf-8") as f:
        original_test_cases = json.load(f)
    
    enhanced_test_cases = []
    for test_case in original_test_cases:
        complaint = test_case['vars']['complaint']
        
        retrieved_context = rag.get_context(complaint, top_k=top_k)
        
        # Create simple test case with just variables, large tests are skipped
        enhanced_test_case = {
            "vars": {
                "complaint": complaint,
                "context": retrieved_context
            }
        }
        
        enhanced_test_cases.append(enhanced_test_case)
    
    return enhanced_test_cases

# Load prompts and test cases
prompts = load_prompts_from_directory(prompts_dir)
test_cases = load_test_cases_from_file(test_cases_file)

# Write Promptfoo config
promptfoo_config = {
    "description": "Customer support evaluation with RAG",
    "providers": [
        "ollama:phi:latest"
    ],
    "prompts": prompts,
    "tests": test_cases
}

with open(promptfoo_output_path, "w", encoding="utf-8") as f:
    yaml.dump(promptfoo_config, f, sort_keys=False, allow_unicode=True)

print(f"✓ Promptfoo config saved to: {promptfoo_output_path}")
print(f"✓ Loaded {len(prompts)} prompts from {prompts_dir}")
print(f"✓ Loaded {len(test_cases)} test cases from {test_cases_file}")
print(f"✓ Enhanced test cases with RAG context (top-{top_k} documents)")
