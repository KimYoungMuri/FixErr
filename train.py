import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config.model_config import TransformerConfig
from models.transformer_model import CodeRepairTransformer
from data.dataset import CodeRepairDataset
from training.trainer import TransformerTrainer

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Create datasets
    train_dataset = CodeRepairDataset("data/processed_codexglue/processed_train.json")
    val_dataset = CodeRepairDataset("data/processed_codexglue/processed_validation.json")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model configuration
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        dropout=0.1,
        layer_norm_eps=1e-12,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
        num_epochs=3
    )
    
    # Initialize model
    model = CodeRepairTransformer(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        dropout=config.dropout,
        layer_norm_eps=config.layer_norm_eps,
        pad_token_id=config.pad_token_id,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id
    )
    
    # Initialize trainer
    trainer = TransformerTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        device=device,
        use_wandb=True
    )
    
    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Train model
    trainer.train(
        num_epochs=config.num_epochs,
        save_dir="checkpoints",
        eval_steps=1000,
        save_steps=5000,
        gradient_accumulation_steps=1
    )

if __name__ == "__main__":
    main() 