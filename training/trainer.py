import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from config.model_config import TransformerConfig
from models.transformer_model import CodeRepairTransformer


class TransformerTrainer:
    """Trainer for the transformer-based code repair model."""
    
    def __init__(
        self,
        model: CodeRepairTransformer,
        config: TransformerConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        use_wandb: bool = True,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_wandb = use_wandb
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=config.num_epochs,
            steps_per_epoch=len(train_dataloader),
            pct_start=0.1,
        )
        
        # Initialize wandb
        if use_wandb:
            wandb.init(
                project="code-repair-transformer",
                config=vars(config),
            )
            wandb.watch(model)
    
    def train(
        self,
        num_epochs: int,
        save_dir: str,
        eval_steps: int = 1000,
        save_steps: int = 5000,
        gradient_accumulation_steps: int = 1,
    ):
        """Train the model."""
        self.model.train()
        global_step = 0
        best_eval_loss = float("inf")
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                leave=False,
            )
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                
                # Update progress bar
                epoch_loss += loss.item() * gradient_accumulation_steps
                progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
                
                # Evaluate
                if global_step % eval_steps == 0 and self.eval_dataloader is not None:
                    eval_loss = self.evaluate()
                    self.model.train()
                    
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        self.save_model(os.path.join(save_dir, "best_model.pt"))
                    
                    if self.use_wandb:
                        wandb.log({
                            "eval_loss": eval_loss,
                            "best_eval_loss": best_eval_loss,
                        }, step=global_step)
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    self.save_model(os.path.join(save_dir, f"checkpoint-{global_step}.pt"))
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        "train_loss": loss.item() * gradient_accumulation_steps,
                        "learning_rate": self.scheduler.get_last_lr()[0],
                    }, step=global_step)
            
            # End of epoch
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_epoch_loss:.4f}")
            
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "avg_epoch_loss": avg_epoch_loss,
                })
    
    def evaluate(self) -> float:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
        
        return total_loss / len(self.eval_dataloader)
    
    def save_model(self, path: str):
        """Save the model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
        }, path)
    
    def load_model(self, path: str):
        """Load the model."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.config = checkpoint["config"] 