import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

from config.model_config import TransformerConfig
from data.dataset import CodeRepairDataModule
from models.transformer_model import CodeRepairTransformer
from training.finetune import Finetuner
from training.pretrain import Pretrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a transformer model for code repair")
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the dataset",
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/codebert-base",
        help="Name of the pretrained model to use",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=50000,
        help="Size of the vocabulary",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=768,
        help="Size of the hidden layers",
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=6,
        help="Number of hidden layers",
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=12,
        help="Number of attention heads",
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm",
    )
    
    # Generation arguments
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Number of beams for beam search",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k for sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p for sampling",
    )
    
    # Other arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save the model",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Whether to use Weights & Biases",
    )
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="Whether to pretrain the model",
    )
    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=5,
        help="Number of pretraining epochs",
    )
    parser.add_argument(
        "--pretrain_output_dir",
        type=str,
        default="pretrained",
        help="Directory to save the pretrained model",
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create model configuration
    config = TransformerConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_length=args.max_length,
        num_beams=args.num_beams,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    
    # Create model
    model = CodeRepairTransformer(config)
    
    # Create data module
    data_module = CodeRepairDataModule(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
    )
    
    # Pretrain if specified
    if args.pretrain:
        logging.info("Starting pretraining...")
        pretrainer = Pretrainer(
            model=model,
            config=config,
            tokenizer=tokenizer,
            output_dir=args.pretrain_output_dir,
            use_wandb=args.use_wandb,
        )
        
        pretrainer.pretrain(
            train_dataloader=data_module.train_dataloader(),
            val_dataloader=data_module.val_dataloader(),
            num_epochs=args.pretrain_epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
        )
        
        # Load best pretrained model
        pretrainer.load_checkpoint("best_model")
    
    # Fine-tune
    logging.info("Starting fine-tuning...")
    finetuner = Finetuner(
        model=model,
        config=config,
        tokenizer=tokenizer,
        output_dir=output_dir,
        use_wandb=args.use_wandb,
    )
    
    finetuner.finetune(
        train_dataloader=data_module.train_dataloader(),
        val_dataloader=data_module.val_dataloader(),
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
    )
    
    # Load best model
    finetuner.load_checkpoint("best_model")
    
    # Test on a sample
    test_dataloader = data_module.test_dataloader()
    for batch in test_dataloader:
        code = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)
        error_msg = tokenizer.decode(batch["error_input_ids"][0], skip_special_tokens=True)
        
        repair = finetuner.generate_repair(
            code=code,
            error_msg=error_msg,
        )
        
        logging.info(f"Code: {code}")
        logging.info(f"Error: {error_msg}")
        logging.info(f"Repair: {repair}")
        break


if __name__ == "__main__":
    main() 