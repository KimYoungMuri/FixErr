import os
import shutil
import gzip
from pathlib import Path
import logging
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url: str, output_path: Path):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def setup_deepfix(data_dir: str):
    """Set up the DeepFix dataset.
    
    Args:
        data_dir: Path to the DeepFix directory
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create necessary directories
    (data_dir / "data").mkdir(exist_ok=True)
    (data_dir / "data" / "iitk-dataset").mkdir(exist_ok=True)
    
    # Download dataset
    logger.info("Downloading DeepFix dataset...")
    dataset_url = "https://www.cse.iitk.ac.in/users/karkare/prutor/prutor-deepfix-09-12-2017.zip"
    zip_path = data_dir / "data" / "prutor-deepfix-09-12-2017.zip"
    download_file(dataset_url, zip_path)
    
    # Extract zip file
    logger.info("Extracting dataset...")
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir / "data" / "temp")
    
    # Move files to correct location
    logger.info("Moving files to correct location...")
    temp_dir = data_dir / "data" / "temp" / "prutor-deepfix-09-12-2017"
    for file in temp_dir.glob("*"):
        shutil.move(str(file), str(data_dir / "data" / "iitk-dataset"))
    
    # Clean up
    logger.info("Cleaning up...")
    shutil.rmtree(data_dir / "data" / "temp")
    zip_path.unlink()
    
    # Extract database
    logger.info("Extracting database...")
    db_gz = data_dir / "data" / "iitk-dataset" / "prutor-deepfix-09-12-2017.db.gz"
    db_path = data_dir / "data" / "prutor-deepfix-09-12-2017.db"
    
    with gzip.open(db_gz, 'rb') as f_in:
        with open(db_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Clean up gzipped database
    db_gz.unlink()
    
    logger.info("Dataset setup complete!")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Set up DeepFix dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to DeepFix directory")
    args = parser.parse_args()
    
    setup_deepfix(args.data_dir)

if __name__ == "__main__":
    main() 