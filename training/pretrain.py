import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from ..config.model_config import TransformerConfig
from ..models.transformer_model import CodeRepairTransformer
from .trainer import TransformerTrainer


class Pretrainer:
    """Pretrain the transformer model using masked language modeling."""
    
    def __init__(
        self,
        model: CodeRepairTransformer,
        config: TransformerConfig,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: Optional[Path] = None,
        use_wandb: bool = False,
    ):
        self.model = model.to(device)
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else Path("pretrained")
        self.use_wandb = use_wandb
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.output_dir / "pretrain.log"),
                logging.StreamHandler(),
            ],
        )
        
        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=config.warmup_steps,
            pct_start=0.1,
        )
        
        # Initialize trainer
        self.trainer = TransformerTrainer(
            model=self.model,
            config=config,
            device=device,
            use_wandb=use_wandb,
        )
    
    def mask_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply masked language modeling to input tokens."""
        # Create mask
        mask = torch.rand(input_ids.shape) < self.config.mlm_probability
        
        # Don't mask special tokens
        special_tokens = [
            self.config.pad_token_id,
            self.config.bos_token_id,
            self.config.eos_token_id,
            self.config.unk_token_id,
        ]
        for token_id in special_tokens:
            mask &= (input_ids != token_id)
        
        # Apply mask
        masked_input_ids = input_ids.clone()
        masked_input_ids[mask] = self.config.unk_token_id
        
        return masked_input_ids, mask
    
    def pretrain(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        num_epochs: int = 10,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        save_steps: int = 1000,
    ):
        """Pretrain the model using masked language modeling."""
        logging.info("Starting pretraining...")
        
        # Training loop
        global_step = 0
        best_val_loss = float("inf")
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            
            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                leave=False,
            )
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Mask tokens
                masked_input_ids, mask = self.mask_tokens(batch["input_ids"])
                batch["input_ids"] = masked_input_ids
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Update weights
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_grad_norm,
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Update progress
                    epoch_loss += loss.item() * gradient_accumulation_steps
                    progress_bar.set_postfix({"loss": loss.item()})
                    
                    # Save checkpoint
                    if global_step % save_steps == 0:
                        self.save_checkpoint(f"checkpoint-{global_step}")
                    
                    global_step += 1
            
            # Log epoch statistics
            avg_loss = epoch_loss / len(train_dataloader)
            logging.info(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
            
            # Evaluate on validation set
            if val_dataloader is not None:
                val_loss = self.evaluate(val_dataloader)
                logging.info(f"Validation Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint("best_model")
        
        # Save final model
        self.save_checkpoint("final_model")
        logging.info("Pretraining completed!")
    
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Evaluate the model on the validation set."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
        
        return total_loss / len(dataloader)
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(
            self.model.state_dict(),
            checkpoint_dir / "model.pt",
        )
        
        # Save optimizer state
        torch.save(
            self.optimizer.state_dict(),
            checkpoint_dir / "optimizer.pt",
        )
        
        # Save scheduler state
        torch.save(
            self.scheduler.state_dict(),
            checkpoint_dir / "scheduler.pt",
        )
        
        # Save configuration
        self.config.save_pretrained(checkpoint_dir)
        
        logging.info(f"Saved checkpoint: {name}")
    
    def load_checkpoint(self, name: str):
        """Load model checkpoint."""
        checkpoint_dir = self.output_dir / name
        
        # Load model state
        self.model.load_state_dict(
            torch.load(checkpoint_dir / "model.pt"),
        )
        
        # Load optimizer state
        self.optimizer.load_state_dict(
            torch.load(checkpoint_dir / "optimizer.pt"),
        )
        
        # Load scheduler state
        self.scheduler.load_state_dict(
            torch.load(checkpoint_dir / "scheduler.pt"),
        )
        
        logging.info(f"Loaded checkpoint: {name}") 