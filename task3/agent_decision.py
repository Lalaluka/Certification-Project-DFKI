import os
import json
import sys
from typing import List
import ollama
sys.path.append('../shared')
from rag_utils import create_rag

# === Config ===
retrieval_threshold = 0.6
prompts_dir = "../task1/prompts"
test_cases_file = "../task1/test_cases.json"

rag = create_rag()

def generate_response(prompt: str) -> str:
    response = ollama.chat(model="phi", messages=[{"role": "user", "content": prompt}])
    return response['message']['content'].strip()

# === System Prompts ===
def load_prompts_from_directory(prompts_dir):
    """Load prompts from the prompts directory"""
    prompts = {}
    
    for filename in os.listdir(prompts_dir):
        if filename.endswith('.txt'):
            prompt_name = filename[:-4]  # Remove .txt extension
            filepath = os.path.join(prompts_dir, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                prompts[prompt_name] = content
    
    return prompts

def load_test_cases_from_file(test_cases_file):
    """Load test cases from JSON file - extract just the complaints"""
    with open(test_cases_file, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    # Extract just the complaints
    complaints = [test_case['vars']['complaint'] for test_case in test_cases]
    return complaints

# === Retrieval Decision ===
def handle_complaint(complaint: str, prompts, prompt_type="empathetic"):
    """Handle complaint using RAG system"""
    retrieval_score = rag.get_retrieval_score(complaint)

    system_prompt = prompts[prompt_type]
    
    if retrieval_score >= retrieval_threshold:
        # Use RAG version - add context to the prompt
        context = rag.get_context(complaint, top_k=2)
        rag_prompt = system_prompt.replace(
            "Customer Complaint: {{complaint}}",
            f"Relevant information from our knowledge base:\n{context}\n\nCustomer Complaint: {{complaint}}"
        )
        prompt = rag_prompt.replace("{{complaint}}", complaint)
        decision = "RAG"
    else:
        # Use prompt-only version
        prompt = system_prompt.replace("{{complaint}}", complaint)
        decision = "Prompt-Only"

    response = generate_response(prompt)

    return {
        "complaint": complaint,
        "prompt_type": prompt_type,
        "decision": decision,
        "retrieval_score": float(round(retrieval_score, 3)),
        "response": response
    }

def main():
    # Load prompts and test cases
    prompts = load_prompts_from_directory(prompts_dir)
    test_complaints = load_test_cases_from_file(test_cases_file)
    
    print(f"Loaded {len(prompts)} prompts: {list(prompts.keys())}")
    print(f"Loaded {len(test_complaints)} test complaints")
    print(f"Using unified RAG system with threshold {retrieval_threshold}")

    results = []
    
    # Test each complaint with each prompt type
    for prompt_type in prompts.keys():
        print(f"\n=== Testing with {prompt_type} prompt ===")
        for complaint in test_complaints:
            result = handle_complaint(complaint, prompts, prompt_type)
            results.append(result)
            print(f"\nComplaint: {complaint}")
            print(f"Prompt: {prompt_type}, Decision: {result['decision']}, Score: {result['retrieval_score']}")
            print(f"Response: {result['response'][:100]}...")

    with open("agent_decision_log.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to agent_decision_log.json ({len(results)} total results)")

if __name__ == "__main__":
    main()
