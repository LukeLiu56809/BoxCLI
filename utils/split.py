from pathlib import Path
import random
from typing import List, Tuple
import shutil
from utils.image import get_image_files

def split_dataset(
    input_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    """
    Split a dataset into training and validation sets
    
    Args:
        input_dir: Directory containing images and annotations
        output_dir: Directory to save split dataset
        train_ratio: Ratio of images to use for training (default: 0.8)
        seed: Random seed for reproducibility (default: 42)
        
    Returns:
        Tuple of (train_files, val_files)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Get all image files
    image_files = get_image_files(input_dir)
    if not image_files:
        raise ValueError(f"No images found in {input_dir}")
    
    # Shuffle files
    random.shuffle(image_files)
    
    # Calculate split index
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files to respective directories
    for img_path in train_files:
        # Copy image
        shutil.copy2(img_path, train_dir / img_path.name)
        # Copy annotation if exists
        ann_path = input_dir / f"{img_path.stem}.txt"
        if ann_path.exists():
            shutil.copy2(ann_path, train_dir / ann_path.name)
    
    for img_path in val_files:
        # Copy image
        shutil.copy2(img_path, val_dir / img_path.name)
        # Copy annotation if exists
        ann_path = input_dir / f"{img_path.stem}.txt"
        if ann_path.exists():
            shutil.copy2(ann_path, val_dir / ann_path.name)
    
    return train_files, val_files 