import os
import json
from pathlib import Path
import logging
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeXGLUEDataConverter:
    def __init__(self, output_dir: str, subset: str = "medium"):
        """Initialize the CodeXGLUE dataset converter.
        
        Args:
            output_dir: Path to save the converted JSON files
            subset: 'small' or 'medium' (default: 'medium')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.subset = subset
        
    def convert(self):
        """Convert the CodeXGLUE code refinement dataset to our JSON format."""
        logger.info(f"Loading CodeXGLUE code refinement dataset ({self.subset}) from HuggingFace...")
        
        # Load the dataset from HuggingFace
        dataset = load_dataset("google/code_x_glue_cc_code_refinement", self.subset)
        
        # Process splits
        splits = {
            'train': dataset['train'],
            'validation': dataset['validation'],
            'test': dataset['test']
        }
        
        for split_name, split_data in splits.items():
            logger.info(f"Processing {split_name} split...")
            processed_data = []
            
            for item in tqdm(split_data):
                processed_item = {
                    "code": item['buggy'],
                    "error_msg": "",  # No error messages in this dataset
                    "fixed_code": item['fixed']
                }
                processed_data.append(processed_item)
            
            # Save to JSON file
            output_file = self.output_dir / f"{split_name}.json"
            with open(output_file, "w") as f:
                json.dump(processed_data, f, indent=2)
            
            logger.info(f"Saved {len(processed_data)} samples to {output_file}")
        
        logger.info("Conversion complete!")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert CodeXGLUE code refinement dataset to JSON format")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save converted JSON files")
    parser.add_argument("--subset", type=str, default="medium", choices=["small", "medium"], help="Subset to use: 'small' or 'medium' (default: 'medium')")
    args = parser.parse_args()
    
    converter = CodeXGLUEDataConverter(args.output_dir, args.subset)
    converter.convert()

if __name__ == "__main__":
    main() 