from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, concatenate_datasets
import torch
import sys
import random
import json
import time
from datetime import datetime
import os

model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# Load datasets
print("## Loading Alpaca dataset ##")
alpaca_dataset = load_dataset("json", data_files="data/alpaca_subset.json", split="train")

print(f"Loaded {len(alpaca_dataset)} Alpaca examples")

# Format Alpaca dataset
def format_alpaca_prompt(example):
    """Format Alpaca dataset examples"""
    return {
        "text": example["text"] + tokenizer.eos_token,
        "source": "alpaca"
    }

formatted_alpaca = alpaca_dataset.map(format_alpaca_prompt, num_proc=1)
print("Formatted Alpaca dataset.")
# Print Example for verification
print("## Example Alpaca Prompt and Output ##")
print(f"Example Alpaca Prompt: {formatted_alpaca[0]['text']}")

## Load HH dataset
print("## Loading HH dataset ##")
hh_dataset = load_dataset("json", data_files="data/hh_subset.json", split="train")

# Reduce HH dataset size to 2000 for faster training
if len(hh_dataset) > 2000:
    hh_dataset = hh_dataset.shuffle(seed=42).select(range(2000))
    print(f"Reduced HH dataset to {len(hh_dataset)} examples for faster training")

print(f"Loaded {len(hh_dataset)} HH examples")

# Format HH dataset 
def format_hh_prompt(example):
    """Formats HH (chosen) examples compatible with Alpaca SFT format."""
    conversation = example["chosen"].strip()
    
    parts = conversation.rsplit("Assistant:", 1)
    
    instruction_context = ""
    final_assistant_response = ""

    if len(parts) == 2:
        instruction_context = parts[0].strip() + "\n\nAssistant:"
        final_assistant_response = parts[1].strip()
    else:
        instruction_context = conversation.strip()
        final_assistant_response = " "
    
    if not final_assistant_response or final_assistant_response.strip() == "":
        final_assistant_response = " "
    
    prefix = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    
    full_prompt_text = f"{prefix}\n\n### Instruction: {instruction_context}\n\n### Response: {final_assistant_response}{tokenizer.eos_token}"

    return {
        "text": full_prompt_text,
        "source": "hh"
    }

print("Formatting HH dataset...")
formatted_hh = hh_dataset.map(format_hh_prompt, num_proc=1)

print("Concatenating datasets...")
formatted_dataset = concatenate_datasets([formatted_alpaca, formatted_hh])

# Tokenize the formatted dataset
def tokenize_example(example):
    """Tokenize the complete text for training"""
    return {
        "input_ids": tokenizer(example["text"], truncation=True, max_length=512, padding="max_length").input_ids,
        "labels": tokenizer(example["text"], truncation=True, max_length=512, padding="max_length").input_ids,
    }

print("Tokenizing dataset...")
tokenized_dataset = formatted_dataset.map(tokenize_example, num_proc=1, remove_columns=formatted_dataset.column_names)

# Create train/validation split
print("Creating train/validation split...")
total_size = len(tokenized_dataset)
val_size = int(total_size * 0.2)  # 20% for validation
train_size = total_size - val_size

# Shuffle and split
shuffled_dataset = tokenized_dataset.shuffle(seed=42)
train_dataset = shuffled_dataset.select(range(train_size))
val_dataset = shuffled_dataset.select(range(train_size, total_size))

print(f"Train dataset: {len(train_dataset)} examples")
print(f"Validation dataset: {len(val_dataset)} examples")

# Hyperparameter search space
HYPERPARAMETER_SPACE = {
    'learning_rate': [1e-5, 5e-5, 1e-4],
    'epochs': [1, 2],
    'batch_size': [4, 8],
    'gradient_accumulation_steps': [8, 16],
    'lora_rank': [8, 16],
    # Is not used but calculated as 2x lora_rank
    'lora_alpha': [16, 32],
    'lora_dropout': [0.05, 0.1],
    'warmup_steps': [50, 100]
}

def generate_random_hyperparams():
    """Generate random hyperparameters from the search space"""
    # First select lora_rank
    lora_rank = random.choice(HYPERPARAMETER_SPACE['lora_rank'])
    
    return {
        'learning_rate': random.choice(HYPERPARAMETER_SPACE['learning_rate']),
        'epochs': random.choice(HYPERPARAMETER_SPACE['epochs']),
        'batch_size': random.choice(HYPERPARAMETER_SPACE['batch_size']),
        'gradient_accumulation_steps': random.choice(HYPERPARAMETER_SPACE['gradient_accumulation_steps']),
        'lora_rank': lora_rank,
        'lora_alpha': lora_rank * 2,  # Always set to 2x lora_rank
        'lora_dropout': random.choice(HYPERPARAMETER_SPACE['lora_dropout']),
        'warmup_steps': random.choice(HYPERPARAMETER_SPACE['warmup_steps'])
    }

def train_model_with_hyperparams(hyperparams, experiment_id):
    """Train a model with specific hyperparameters"""
    print(f"\n=== Training Model {experiment_id} ===")
    print(f"Hyperparameters: {hyperparams}")
    
    # Create fresh model for this experiment
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float32,
    )
    
    # Create LoRA config with hyperparameters
    lora_config = LoraConfig(
        r=hyperparams['lora_rank'],
        lora_alpha=hyperparams['lora_alpha'],
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=hyperparams['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(base_model, lora_config)
    
    # Output directory for this experiment
    output_dir = f"experiments/model_{experiment_id}"
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=hyperparams['batch_size'],
        gradient_accumulation_steps=hyperparams['gradient_accumulation_steps'],
        learning_rate=hyperparams['learning_rate'],
        num_train_epochs=hyperparams['epochs'],
        warmup_steps=hyperparams['warmup_steps'],
        
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        bf16=False,
        fp16=False,
        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
        max_grad_norm=1.0,
        dataloader_pin_memory=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    # Train model
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    # Get final validation loss
    eval_results = trainer.evaluate()
    val_loss = eval_results['eval_loss']
    
    # Save the final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Clean up memory
    del model
    del base_model
    del trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return {
        'val_loss': val_loss,
        'training_time': training_time,
        'output_dir': output_dir
    }

def random_hyperparameter_search(n_trials=5):
    """Perform random hyperparameter search with unique combinations"""
    print(f"\n{'='*50}")
    print(f"Starting Random Hyperparameter Search")
    print(f"Number of trials: {n_trials}")
    print(f"{'='*50}")
    
    results = []
    used_combinations = set()  # Track used combinations
    
    # Create experiments directory
    os.makedirs("experiments", exist_ok=True)
    
    trial = 0
    attempts = 0
    max_attempts = n_trials * 10  # Prevent infinite loop
    
    while trial < n_trials and attempts < max_attempts:
        attempts += 1
        try:
            # Generate random hyperparameters
            hyperparams = generate_random_hyperparams()
            
            # Create a hashable key from hyperparameters for uniqueness check
            param_key = tuple(sorted(hyperparams.items()))
            
            # Check if this combination has been used before
            if param_key in used_combinations:
                print(f"⚠️  Skipping duplicate combination (attempt {attempts})")
                continue
            
            # Mark this combination as used
            used_combinations.add(param_key)
            trial += 1
            
            print(f"\n=== Training Model {trial} (attempt {attempts}) ===")
            print(f"Unique combinations so far: {len(used_combinations)}")
            
            # Train model
            training_results = train_model_with_hyperparams(hyperparams, trial)
            
            # Store results
            experiment_result = {
                'experiment_id': trial,
                'hyperparameters': hyperparams,
                'val_loss': training_results['val_loss'],
                'training_time': training_results['training_time'],
                'output_dir': training_results['output_dir'],
                'attempt_number': attempts
            }
            
            results.append(experiment_result)
            
            print(f"✅ Experiment {trial} completed:")
            print(f"   Validation Loss: {training_results['val_loss']:.4f}")
            print(f"   Training Time: {training_results['training_time']:.1f}s")
            
            # Save intermediate results
            with open('hyperparameter_search_results.json', 'w') as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            print(f"❌ Experiment {trial} failed: {e}")
            # Don't increment trial counter for failed experiments
            continue
    
    if attempts >= max_attempts:
        print(f"⚠️  Reached maximum attempts ({max_attempts}). Completed {trial} unique experiments.")
    
    print(f"\nCompleted {trial} experiments with {len(used_combinations)} unique combinations")
    return results

def analyze_results(results):
    """Analyze and display results"""
    print(f"\n{'='*50}")
    print("HYPERPARAMETER SEARCH RESULTS")
    print(f"{'='*50}")
    
    if not results:
        print("No successful experiments!")
        return
    
    # Sort by validation loss (lower is better)
    sorted_results = sorted(results, key=lambda x: x['val_loss'])
    
    print(f"\nTop 3 Models (by validation loss):")
    for i, result in enumerate(sorted_results[:3]):
        print(f"\n{i+1}. Experiment {result['experiment_id']}:")
        print(f"   Validation Loss: {result['val_loss']:.4f}")
        print(f"   Training Time: {result['training_time']:.1f}s")
        print(f"   Hyperparameters:")
        for param, value in result['hyperparameters'].items():
            print(f"     {param}: {value}")
        print(f"   Model saved in: {result['output_dir']}")
    
    # Best model
    best_model = sorted_results[0]
    print(f"\n🏆 BEST MODEL: Experiment {best_model['experiment_id']}")
    print(f"   Final validation loss: {best_model['val_loss']:.4f}")
    print(f"   Model location: {best_model['output_dir']}")
    
    return best_model

# Run hyperparameter search
random.seed(42)  # For reproducibility
search_results = random_hyperparameter_search(n_trials=10)

# Analyze results
best_model = analyze_results(search_results)

print(f"\n{'='*50}")
print("Hyperparameter search completed!")
print(f"Results saved to: hyperparameter_search_results.json")
print(f"Best model saved to: {best_model['output_dir'] if best_model else 'No successful models'}")
print(f"{'='*50}")
