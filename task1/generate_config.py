#!/usr/bin/env python3

import os
import yaml
import json
from pathlib import Path

def load_prompts_from_directory(prompts_dir="prompts"):
    """Load all .txt files from prompts directory"""
    prompts = []
    prompts_path = Path(prompts_dir)
    
    if prompts_path.exists():
        for prompt_file in prompts_path.glob("*.txt"):
            prompt_name = prompt_file.stem
            prompts.append({
                "id": prompt_name,
                "raw": f"file://{prompt_file}"
            })
    
    return prompts

def load_test_cases_from_file(test_file="test_cases.json"):
    """Load test cases from JSON file"""
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            return json.load(f)
    return []

def generate_config():
    """Generate complete promptfoo configuration"""
    
    # Load dynamic prompts
    prompts = load_prompts_from_directory()
    
    # Load test cases
    test_cases = load_test_cases_from_file()
    
    # Base configuration
    config = {
        "description": "Customer support prompt evaluation (dynamically generated)",
        "providers": [
            "ollama:mistral:latest",
            "ollama:phi:latest"
        ],
        "prompts": prompts,
        "tests": test_cases
    }
    
    return config

def main():
    """Generate and save promptfoo config"""
    config = generate_config()
    
    # Save to YAML with proper Unicode handling
    with open("promptfooconfig.yaml", "w", encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"✓ Generated promptfoo config with {len(config['prompts'])} prompts")
    print(f"✓ Loaded {len(config['tests'])} test cases")
    print("✓ Config saved to promptfooconfig.yaml")

if __name__ == "__main__":
    main()
