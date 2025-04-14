from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image

def load_image(image_path: Path) -> np.ndarray:
    """
    Load an image from file
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Numpy array containing the image
    """
    return cv2.imread(str(image_path))

def save_image(image: np.ndarray, output_path: Path) -> None:
    """
    Save an image to file
    
    Args:
        image: Numpy array containing the image
        output_path: Path to save the image
    """
    cv2.imwrite(str(output_path), image)

def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resize an image to specified dimensions
    
    Args:
        image: Numpy array containing the image
        width: Target width
        height: Target height
    
    Returns:
        Resized image as numpy array
    """
    return cv2.resize(image, (width, height))

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values to [0, 1]
    
    Args:
        image: Numpy array containing the image
    
    Returns:
        Normalized image as numpy array
    """
    return image.astype(np.float32) / 255.0

def get_image_size(image: np.ndarray) -> Tuple[int, int]:
    """
    Get image dimensions
    
    Args:
        image: Numpy array containing the image
    
    Returns:
        Tuple of (height, width)
    """
    return image.shape[:2]

def get_supported_image_extensions() -> List[str]:
    """
    Get list of supported image file extensions
    
    Returns:
        List of supported extensions
    """
    return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

def is_image_file(file_path: Path) -> bool:
    """
    Check if file is an image
    
    Args:
        file_path: Path to the file
    
    Returns:
        True if file is an image, False otherwise
    """
    return file_path.suffix.lower() in get_supported_image_extensions()

def get_image_files(directory: Path) -> List[Path]:
    """
    Get all image files in a directory
    
    Args:
        directory: Directory to search
    
    Returns:
        List of image file paths
    """
    return [f for f in directory.glob('*') if is_image_file(f)] 