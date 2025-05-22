import os
import json
import sqlite3
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepFixDataConverter:
    def __init__(self, data_dir: str, output_dir: str):
        """Initialize the DeepFix dataset converter.
        
        Args:
            data_dir: Path to the DeepFix dataset directory
            output_dir: Path to save the converted JSON files
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths to dataset files
        self.db_path = self.data_dir / "data" / "prutor-deepfix-09-12-2017.db"
        self.bins_path = self.data_dir / "data" / "iitk-dataset" / "bins.npy"
        self.validation_users_path = self.data_dir / "data" / "iitk-dataset" / "validation_users.npy"
        
    def load_bins(self) -> np.ndarray:
        """Load the bins.npy file containing problem IDs."""
        if not self.bins_path.exists():
            raise FileNotFoundError(f"bins.npy not found at {self.bins_path}")
        return np.load(self.bins_path, allow_pickle=True)
    
    def load_validation_users(self) -> dict:
        """Load the validation_users.npy file as a dict."""
        if not self.validation_users_path.exists():
            raise FileNotFoundError(f"validation_users.npy not found at {self.validation_users_path}")
        return np.load(self.validation_users_path, allow_pickle=True).item()
    
    def process_code_sample(self, code: str, error: str) -> Dict:
        """Process a single code sample into our format.
        
        Args:
            code: The buggy code
            error: The error message
            
        Returns:
            Dictionary in our required format
        """
        return {
            "code": code,
            "error_msg": error,
            "fixed_code": code  # For now, we'll use the same code as fixed code
        }
    
    def convert(self):
        """Convert the DeepFix dataset to our JSON format."""
        logger.info("Loading DeepFix dataset...")
        
        # Load problem IDs and validation users
        bins = self.load_bins()
        validation_users = self.load_validation_users()
        
        logger.info(f"Loaded {len(bins)} problem bins")
        logger.info(f"Loaded {len(validation_users)} validation user lists (per problem)")
        
        # Process all code samples
        logger.info("Processing code samples...")
        train_data = []
        val_data = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get all problems
            problem_list = []
            for bin_ in bins:
                for problem_id in bin_:
                    problem_list.append(problem_id)
            
            logger.info(f"Found {len(problem_list)} unique problems")
            
            # Process each problem
            for problem_id in tqdm(problem_list):
                # Get all code samples for this problem
                query = """
                    SELECT user_id, code, error
                    FROM Code
                    WHERE problem_id = ? AND errorcount > 0
                """
                
                for row in cursor.execute(query, (problem_id,)):
                    user_id, code, error = row
                    
                    # Skip if no error message
                    if not error:
                        continue
                    
                    sample = self.process_code_sample(code, error)
                    
                    # Assign to validation if user_id is in validation_users[problem_id], else train
                    if problem_id in validation_users and user_id in validation_users[problem_id]:
                        val_data.append(sample)
                    else:
                        train_data.append(sample)
        
        logger.info(f"Found {len(train_data)} training samples")
        logger.info(f"Found {len(val_data)} validation samples")
        
        # If we have no validation data, do a random split of training data
        if len(val_data) == 0:
            logger.info("No validation data found. Splitting training data randomly...")
            np.random.shuffle(train_data)
            val_size = int(len(train_data) * 0.1)  # 10% for validation
            test_size = int(len(train_data) * 0.1)  # 10% for test
            val_data = train_data[:val_size]
            test_data = train_data[val_size:val_size + test_size]
            train_data = train_data[val_size + test_size:]
        else:
            # Split validation data into validation and test sets
            np.random.shuffle(val_data)
            val_size = int(len(val_data) * 0.5)
            test_data = val_data[val_size:]
            val_data = val_data[:val_size]
        
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
    parser = argparse.ArgumentParser(description="Convert DeepFix dataset to JSON format")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to DeepFix dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save converted JSON files")
    args = parser.parse_args()
    
    converter = DeepFixDataConverter(args.data_dir, args.output_dir)
    converter.convert()

if __name__ == "__main__":
    main() 