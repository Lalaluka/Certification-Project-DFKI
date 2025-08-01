#!/usr/bin/env python3
"""
Generate response files for BERTScore evaluation from different model configurations.
This script creates responses for: prompt_only, rag, fine_tuned, and agent variants.
"""

import json
import os
import sys
import ollama
import numpy as np
import torch
from typing import Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Add shared utilities to path
sys.path.append('../shared')
from rag_utils import create_rag

# Configuration
RESPONSES_DIR = "responses"
TEST_CASES_FILE = "../task1/test_cases.json"
FRIENDLY_PROMPT_FILE = "../task1/prompts/friendly.txt"
FINETUNED_MODEL_PATH = "../task4/experiments/model_X"

# Model configurations
OLLAMA_MODEL = "phi"
BASE_MODEL_NAME = "microsoft/phi-2"

# RAG threshold from Task 3
RETRIEVAL_THRESHOLD = 0.6

def load_test_cases():
    """Extract complaints from test cases"""
    with open(TEST_CASES_FILE, 'r') as f:
        test_cases = json.load(f)
    
    complaints = []
    for case in test_cases:
        if 'vars' in case and 'complaint' in case['vars']:
            complaints.append(case['vars']['complaint'])
    
    return complaints

def load_friendly_prompt():
    """Load the friendly prompt template"""
    with open(FRIENDLY_PROMPT_FILE, 'r') as f:
        return f.read().strip()

class ResponseGenerator:
    def __init__(self):
        self.friendly_prompt = load_friendly_prompt()
        
        # Setup RAG - must be available
        self.setup_rag()
        
        # Setup fine-tuned model - must be available
        self.setup_finetuned_model()
    
    def setup_rag(self):
        """Initialize unified RAG system"""
        print("Setting up unified RAG...")
        self.rag = create_rag()
        print(f"✅ Unified RAG setup complete")
    
    def setup_finetuned_model(self):
        """Load fine-tuned model - must succeed"""
        print("Loading fine-tuned model...")
        
        # Load tokenizer
        self.finetuned_tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
        if self.finetuned_tokenizer.pad_token is None:
            self.finetuned_tokenizer.pad_token = self.finetuned_tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        # Load PEFT model
        self.finetuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_PATH)
        self.finetuned_model.eval()
        
        print("✅ Fine-tuned model loaded successfully")
    
    def get_relevant_context(self, query, top_k=2):
        """Get relevant context using unified RAG"""
        return self.rag.get_context(query, top_k=top_k)
    
    def generate_prompt_only_response(self, complaint):
        """Generate response using only the friendly prompt"""
        prompt = self.friendly_prompt.replace("{{complaint}}", complaint)
        
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content'].strip()
    
    def generate_rag_response(self, complaint):
        """Generate response using RAG (prompt + context)"""
        # Get relevant context
        context = self.get_relevant_context(complaint)
        
        # Create enhanced prompt
        if context:
            enhanced_prompt = f"""Context Information:
{context}

{self.friendly_prompt.replace("{{complaint}}", complaint)}

Use the context information above to provide a more accurate and helpful response."""
        else:
            enhanced_prompt = self.friendly_prompt.replace("{{complaint}}", complaint)
        
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": enhanced_prompt}]
        )
        return response['message']['content'].strip()
    
    def generate_finetuned_response(self, complaint):
        """Generate response using fine-tuned model"""
        prompt = self.friendly_prompt.replace("{{complaint}}", complaint)
        
        inputs = self.finetuned_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.finetuned_model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.finetuned_tokenizer.eos_token_id
            )
        
        response = self.finetuned_tokenizer.decode(
            outputs[0][len(inputs['input_ids'][0]):], 
            skip_special_tokens=True
        )
        return response.strip()
    
    def generate_agent_response(self, complaint):
        """Generate response using fine-tuned model with intelligent RAG decision based on retrieval score"""
        # Get retrieval score using unified RAG (same logic as Task 3)
        retrieval_score = self.rag.get_retrieval_score(complaint)
        
        print(f"    Agent decision - Retrieval score: {retrieval_score:.3f}, Threshold: {RETRIEVAL_THRESHOLD}")
        
        # Decision: Add RAG context if score is above threshold
        if retrieval_score >= RETRIEVAL_THRESHOLD:
            # High relevance -> Use fine-tuned model + RAG context
            context = self.rag.get_context(complaint, top_k=2)
            if context:
                enhanced_prompt = f"""Context Information:
{context}

{self.friendly_prompt.replace("{{complaint}}", complaint)}

Use the context information above to provide a more accurate and helpful response."""
                print(f"    -> Using: Fine-tuned model + RAG")
            else:
                enhanced_prompt = self.friendly_prompt.replace("{{complaint}}", complaint)
                print(f"    -> Using: Fine-tuned model only (no context found)")
        else:
            # Low relevance -> Use fine-tuned model without RAG
            enhanced_prompt = self.friendly_prompt.replace("{{complaint}}", complaint)
            print(f"    -> Using: Fine-tuned model only (low relevance)")
        
        # Generate response using fine-tuned model
        inputs = self.finetuned_tokenizer(enhanced_prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.finetuned_model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.finetuned_tokenizer.eos_token_id
            )
        
        response = self.finetuned_tokenizer.decode(
            outputs[0][len(inputs['input_ids'][0]):], 
            skip_special_tokens=True
        )
        return response.strip()

def main():
    """Main function to generate all response variants"""
    print("🚀 Generating responses for evaluation...")
    
    # Create responses directory
    os.makedirs(RESPONSES_DIR, exist_ok=True)
    
    # Load test cases
    complaints = load_test_cases()
    print(f"📝 Loaded {len(complaints)} test complaints")
    
    # Initialize response generator
    generator = ResponseGenerator()
    
    # Generate responses for each variant
    variants = {
        "prompt_only": generator.generate_prompt_only_response,
        "rag": generator.generate_rag_response,
        "fine_tuned": generator.generate_finetuned_response,
        "agent": generator.generate_agent_response
    }
    
    for variant_name, variant_func in variants.items():
        print(f"\n🔄 Generating {variant_name} responses...")
        
        responses = []
        for i, complaint in enumerate(complaints):
            print(f"  [{i+1}/{len(complaints)}] Processing: {complaint[:50]}...")
            
            response = variant_func(complaint)
            responses.append({
                "complaint": complaint,
                "response": response
            })
        
        # Save responses
        output_file = os.path.join(RESPONSES_DIR, f"{variant_name}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(responses, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved {len(responses)} responses to {output_file}")
    
    print(f"\n🎉 All response variants generated successfully!")
    print(f"📁 Files created in {RESPONSES_DIR}/:")
    for variant in variants.keys():
        print(f"   - {variant}.json")
    
    print(f"\n🔍 Ready for evaluation!")

if __name__ == "__main__":
    main()
