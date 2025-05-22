import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from preprocessing.code_processor import CodeProcessor


class CodeRepairDataset(Dataset):
    """Dataset for code repair using the CodeXGLUE format."""
    
    def __init__(self, data_path: str, max_length: int = 512):
        """Initialize the dataset.
        
        Args:
            data_path: Path to the JSON file containing the processed data
            max_length: Maximum sequence length for input and target sequences
        """
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.max_length = max_length
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Dictionary containing the sample's features
        """
        sample = self.data[idx]
        
        # Convert lists back to tensors and ensure correct dimensions
        buggy_features = {
            'input_ids': torch.tensor(sample['buggy']['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(sample['buggy']['attention_mask'], dtype=torch.long)
        }
        
        fixed_features = {
            'input_ids': torch.tensor(sample['fixed']['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(sample['fixed']['attention_mask'], dtype=torch.long)
        }
        
        # Ensure all tensors are always 1D (seq_length,)
        for features in [buggy_features, fixed_features]:
            for key in features:
                features[key] = features[key].view(-1)  # Always 1D
                if features[key].shape[0] > self.max_length:
                    features[key] = features[key][:self.max_length]
                elif features[key].shape[0] < self.max_length:
                    padding = torch.zeros(
                        self.max_length - features[key].shape[0],
                        dtype=features[key].dtype
                    )
                    features[key] = torch.cat([features[key], padding])
        
        return {
            'input_ids': buggy_features['input_ids'],
            'attention_mask': buggy_features['attention_mask'],
            'labels': fixed_features['input_ids'],
            'decoder_attention_mask': fixed_features['attention_mask']
        }


class CodeRepairDataModule:
    """Data module for code repair tasks."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        max_code_length: int = 512,
        max_error_length: int = 128,
        use_ast: bool = True,
        use_data_flow: bool = True,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_code_length = max_code_length
        self.max_error_length = max_error_length
        self.use_ast = use_ast
        self.use_data_flow = use_data_flow
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Initialize datasets
        self.train_dataset = CodeRepairDataset(
            data_path=self.data_dir / "train.json",
            max_length=self.max_code_length
        )
        
        self.val_dataset = CodeRepairDataset(
            data_path=self.data_dir / "val.json",
            max_length=self.max_code_length
        )
        
        self.test_dataset = CodeRepairDataset(
            data_path=self.data_dir / "test.json",
            max_length=self.max_code_length
        )
    
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the training dataloader."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the validation dataloader."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the test dataloader."""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        ) 