from pathlib import Path
from typing import List, Tuple
import numpy as np
from annotator import BoundingBox

class YOLOConverter:
    @staticmethod
    def save_annotations(annotations: List[BoundingBox], output_path: Path, img_width: int, img_height: int):
        """Save annotations in YOLO format"""
        with open(output_path, 'w') as f:
            for box in annotations:
                x_center, y_center, box_width, box_height = box.to_yolo(img_width, img_height)
                f.write(f"{box.class_id} {x_center} {y_center} {box_width} {box_height}\n")

    @staticmethod
    def load_annotations(file_path: Path, img_width: int, img_height: int) -> List[BoundingBox]:
        """Load annotations from YOLO format file"""
        annotations = []
        with open(file_path, 'r') as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                # Convert from YOLO format to pixel coordinates
                x_center *= img_width
                y_center *= img_height
                width *= img_width
                height *= img_height
                
                x1 = int(x_center - width/2)
                y1 = int(y_center - height/2)
                x2 = int(x_center + width/2)
                y2 = int(y_center + height/2)
                
                annotations.append(BoundingBox(x1, y1, x2, y2, int(class_id)))
        return annotations

    @staticmethod
    def create_dataset_structure(output_dir: Path, classes: List[str]):
        """Create YOLO dataset structure"""
        # Create necessary directories
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        # Create classes.txt file
        with open(output_dir / "classes.txt", 'w') as f:
            for class_name in classes:
                f.write(f"{class_name}\n") 