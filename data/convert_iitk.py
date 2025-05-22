import os
import json
import tarfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IITKDataConverter:
    def __init__(self, data_dir: str, output_dir: str):
        """Initialize the IITK dataset converter.
        
        Args:
            data_dir: Path to the IITK dataset directory
            output_dir: Path to save the converted JSON files
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths to dataset files
        self.bins_path = self.data_dir / "bins.npy"
        self.problem_statements_path = self.data_dir / "problem_statements.tar.gz"
        self.validation_users_path = self.data_dir / "validation_users.npy"
        
    def load_bins(self) -> np.ndarray:
        """Load the bins.npy file containing code samples."""
        if not self.bins_path.exists():
            raise FileNotFoundError(f"bins.npy not found at {self.bins_path}")
        return np.load(self.bins_path, allow_pickle=True)
    
    def load_validation_users(self) -> np.ndarray:
        """Load the validation_users.npy file."""
        if not self.validation_users_path.exists():
            raise FileNotFoundError(f"validation_users.npy not found at {self.validation_users_path}")
        return np.load(self.validation_users_path, allow_pickle=True)
    
    def extract_problem_statements(self) -> Dict[str, str]:
        """Extract problem statements from the tar.gz file."""
        if not self.problem_statements_path.exists():
            raise FileNotFoundError(f"problem_statements.tar.gz not found at {self.problem_statements_path}")
        
        problem_statements = {}
        with tarfile.open(self.problem_statements_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.name.endswith('.txt'):
                    f = tar.extractfile(member)
                    if f is not None:
                        problem_id = os.path.splitext(os.path.basename(member.name))[0]
                        problem_statements[problem_id] = f.read().decode('utf-8')
        return problem_statements
    
    def process_code_sample(self, sample: str) -> Dict:
        """Process a single code sample into our format.
        
        Args:
            sample: String containing code sample
            
        Returns:
            Dictionary in our required format:
            {
                "code": str,
                "error_msg": str,
                "fixed_code": str
            }
        """
        # For now, we'll use the same code as both input and fixed code
        # since we don't have error messages or fixed versions in the IITK dataset
        return {
            "code": sample,
            "error_msg": "",  # No error messages in IITK dataset
            "fixed_code": sample  # Using same code as fixed version for now
        }
    
    def split_data(self, data: List[Dict], validation_users: np.ndarray) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data into train, validation, and test sets.
        
        Args:
            data: List of processed code samples
            validation_users: Array of user IDs for validation set
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Since we don't have user IDs in the IITK dataset,
        # we'll do a simple random split
        np.random.shuffle(data)
        total_samples = len(data)
        train_size = int(0.8 * total_samples)
        val_size = int(0.1 * total_samples)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        return train_data, val_data, test_data
    
    def convert(self):
        """Convert the IITK dataset to our JSON format."""
        logger.info("Loading IITK dataset...")
        
        # Load data
        bins = self.load_bins()
        validation_users = self.load_validation_users()
        problem_statements = self.extract_problem_statements()
        
        # Process all code samples
        logger.info("Processing code samples...")
        processed_data = []
        for bin_data in tqdm(bins):
            for sample in bin_data:
                processed_sample = self.process_code_sample(sample)
                processed_data.append(processed_sample)
        
        # Split data
        logger.info("Splitting data into train/val/test sets...")
        train_data, val_data, test_data = self.split_data(processed_data, validation_users)
        
        # Save to JSON files
        logger.info("Saving processed data...")
        with open(self.output_dir / "train.json", "w") as f:
            json.dump(train_data, f, indent=2)
        with open(self.output_dir / "val.json", "w") as f:
            json.dump(val_data, f, indent=2)
        with open(self.output_dir / "test.json", "w") as f:
            json.dump(test_data, f, indent=2)
        
        logger.info(f"Conversion complete. Data saved to {self.output_dir}")
        logger.info(f"Train samples: {len(train_data)}")
        logger.info(f"Validation samples: {len(val_data)}")
        logger.info(f"Test samples: {len(test_data)}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert IITK dataset to JSON format")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to IITK dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save converted JSON files")
    args = parser.parse_args()
    
    converter = IITKDataConverter(args.data_dir, args.output_dir)
    converter.convert()

if __name__ == "__main__":
    main() 