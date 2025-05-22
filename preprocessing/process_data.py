import json
import torch
from transformers import AutoTokenizer
from .code_processor import CodeProcessor

def tensor_to_list(tensor_dict):
    """Convert dictionary of tensors to dictionary of lists."""
    return {k: v.tolist() if isinstance(v, torch.Tensor) else v 
            for k, v in tensor_dict.items()}

# Load tokenizer (adjust model name as needed)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Initialize processor
processor = CodeProcessor(tokenizer)

# Process training data
print("Processing training data...")
with open("data/processed_codexglue/train.json") as f:
    train_samples = json.load(f)

# Process each sample and collect features
processed_train_features = []
for sample in train_samples:
    # Process both buggy and fixed code
    buggy_features = processor.process_code(sample['code'])
    fixed_features = processor.process_code(sample['fixed_code'])
    
    # Convert tensors to lists
    buggy_features = tensor_to_list(buggy_features)
    fixed_features = tensor_to_list(fixed_features)
    
    # Combine features
    combined_features = {
        'buggy': buggy_features,
        'fixed': fixed_features,
        'error_msg': sample['error_msg']
    }
    processed_train_features.append(combined_features)

# Save processed training features
with open("data/processed_codexglue/processed_train.json", "w") as f:
    json.dump(processed_train_features, f)

print("Processing validation data...")
# Process validation data
with open("data/processed_codexglue/validation.json") as f:
    val_samples = json.load(f)

# Process each sample and collect features
processed_val_features = []
for sample in val_samples:
    # Process both buggy and fixed code
    buggy_features = processor.process_code(sample['code'])
    fixed_features = processor.process_code(sample['fixed_code'])
    
    # Convert tensors to lists
    buggy_features = tensor_to_list(buggy_features)
    fixed_features = tensor_to_list(fixed_features)
    
    # Combine features
    combined_features = {
        'buggy': buggy_features,
        'fixed': fixed_features,
        'error_msg': sample['error_msg']
    }
    processed_val_features.append(combined_features)

# Save processed validation features
with open("data/processed_codexglue/processed_validation.json", "w") as f:
    json.dump(processed_val_features, f)

print("Preprocessing completed. Processed features saved to processed_train.json and processed_validation.json.") 