import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import json
import os
from tqdm import tqdm
import time
import shutil

class BoundingBox:
    def __init__(self, x1: int, y1: int, x2: int, y2: int, class_id: int):
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
        self.class_id = class_id

    def to_yolo(self, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """Convert to YOLO format (center_x, center_y, width, height) normalized to [0,1]"""
        width = self.x2 - self.x1
        height = self.y2 - self.y1
        center_x = (self.x1 + width/2) / img_width
        center_y = (self.y1 + height/2) / img_height
        norm_width = width / img_width
        norm_height = height / img_height
        return (center_x, center_y, norm_width, norm_height)

    def to_coco(self, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """Convert to COCO format [x, y, width, height]"""
        return (self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

class ImageAnnotator:
    def __init__(self, image_path: Path, classes: List[str], output_path: Path, format: str, current_frame: int, total_frames: int):
        self.image_path = image_path
        self.classes = classes
        self.output_path = output_path
        self.format = format
        self.boxes: List[BoundingBox] = []
        self.current_box: Optional[BoundingBox] = None
        self.drawing = False
        self.current_class = 0
        self.window_name = "Image Annotation"
        self.current_frame = current_frame
        self.total_frames = total_frames
        
        # Load and cache the image
        self.original_image = cv2.imread(str(image_path))
        if self.original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        self.display_image = self.original_image.copy()
        
        # Set up the window
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        # Performance optimization variables
        self.last_update_time = 0
        self.update_threshold = 0.016  # ~60 FPS
        self.needs_update = True
        
        # Load existing annotations if they exist
        if self.output_path.exists():
            self._load_annotations()

    def _load_annotations(self):
        """Load existing annotations from file"""
        if self.format == 'yolo':
            self._load_yolo()
        elif self.format == 'coco':
            self._load_coco()

    def _load_yolo(self):
        """Load annotations in YOLO format"""
        height, width = self.original_image.shape[:2]
        with open(self.output_path, 'r') as f:
            for line in f:
                class_id, center_x, center_y, norm_width, norm_height = map(float, line.strip().split())
                # Convert from YOLO format to pixel coordinates
                x1 = int((center_x - norm_width/2) * width)
                y1 = int((center_y - norm_height/2) * height)
                x2 = int((center_x + norm_width/2) * width)
                y2 = int((center_y + norm_height/2) * height)
                self.boxes.append(BoundingBox(x1, y1, x2, y2, int(class_id)))

    def _load_coco(self):
        """Load annotations in COCO format"""
        with open(self.output_path, 'r') as f:
            coco_ann = json.load(f)
            for ann in coco_ann['annotations']:
                x, y, w, h = ann['bbox']
                self.boxes.append(BoundingBox(int(x), int(y), int(x + w), int(y + h), ann['category_id']))

    def _mouse_callback(self, event, x, y, flags, param):
        current_time = time.time()
        if current_time - self.last_update_time < self.update_threshold:
            return
            
        self.last_update_time = current_time
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_box = BoundingBox(x, y, x, y, self.current_class)
            self.needs_update = True
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            if self.current_box:
                self.current_box.x2 = x
                self.current_box.y2 = y
                self.needs_update = True
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.current_box:
                # Only add box if it has non-zero area
                if abs(self.current_box.x2 - self.current_box.x1) > 5 and abs(self.current_box.y2 - self.current_box.y1) > 5:
                    self.boxes.append(self.current_box)
                self.current_box = None
                self.needs_update = True
                
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove the last box
            if self.boxes:
                self.boxes.pop()
                self.needs_update = True
                
        elif event == cv2.EVENT_MBUTTONDOWN:
            # Cycle through classes
            self.current_class = (self.current_class + 1) % len(self.classes)
            self.needs_update = True

    def _update_display(self):
        """Update the display image with current boxes"""
        if not self.needs_update:
            return
            
        # Reset to original image
        self.display_image = self.original_image.copy()
        
        # Draw existing boxes
        for box in self.boxes:
            self._draw_box(box, (0, 255, 0))
            
        # Draw current box if exists
        if self.current_box:
            self._draw_box(self.current_box, (0, 0, 255))
            
        # Update window title with current class and frame info
        class_name = self.classes[self.current_class]
        window_title = f"BoxCLI - Class: {class_name} | Frame: {self.current_frame}/{self.total_frames}"
        cv2.setWindowTitle(self.window_name, window_title)
        
        cv2.imshow(self.window_name, self.display_image)
        self.needs_update = False

    def _draw_box(self, box: BoundingBox, color: Tuple[int, int, int]):
        """Draw a single box with class label"""
        # Draw rectangle
        cv2.rectangle(self.display_image, (box.x1, box.y1), (box.x2, box.y2), color, 4)
        
        # Draw class name
        class_name = self.classes[box.class_id]
        # Calculate text size
        font_scale = 1.0
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Ensure text doesn't go off screen
        text_x = max(0, min(box.x1, self.display_image.shape[1] - text_width))
        text_y = max(text_height, box.y1 - 10)
        
        # Draw text with black outline
        cv2.putText(self.display_image, class_name, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(self.display_image, class_name, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    def run(self):
        """Run the annotation interface
        
        Returns:
            bool: True if user wants to continue, False if user wants to quit
        """
        self._update_display()
        
        try:
            while True:
                if self.needs_update:
                    self._update_display()
                
                # Use a shorter wait time and handle key press immediately
                key = cv2.waitKeyEx(1) & 0xFF
                if key == 0xFF:  # No key pressed
                    continue
                    
                # Clear any buffered key presses
                while cv2.waitKeyEx(1) & 0xFF != 0xFF:
                    pass
                
                if key == ord('q'):  # Save and quit
                    self.save_annotations()
                    print(f"Annotations saved to {self.output_path}")
                    return False
                elif key == ord('n'):  # Next class
                    self.current_class = (self.current_class + 1) % len(self.classes)
                    self.needs_update = True
                elif key == ord('p'):  # Previous class
                    self.current_class = (self.current_class - 1) % len(self.classes)
                    self.needs_update = True
                elif key == ord('d'):  # Delete last box
                    if self.boxes:
                        self.boxes.pop()
                        self.needs_update = True
                elif key == ord('z'):  # Go back
                    self.save_annotations()
                    return 'back'
                elif key == ord('x'):  # Go forward
                    self.save_annotations()
                    return 'forward'
        finally:
            cv2.destroyAllWindows()

    def save_annotations(self):
        """Save annotations in the specified format"""
        if self.format == 'yolo':
            self._save_yolo()
        elif self.format == 'coco':
            self._save_coco()
        else:
            raise ValueError(f"Unsupported format: {self.format}")

    def _save_yolo(self):
        """Save annotations in YOLO format"""
        height, width = self.original_image.shape[:2]
        lines = []
        for box in self.boxes:
            center_x, center_y, norm_width, norm_height = box.to_yolo(width, height)
            lines.append(f"{box.class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}")
        
        with open(self.output_path, 'w') as f:
            f.write('\n'.join(lines))

    def _save_coco(self):
        """Save annotations in COCO format"""
        height, width = self.original_image.shape[:2]
        image_id = int(self.image_path.stem)
        
        # Create COCO annotation
        coco_ann = {
            "images": [{
                "id": image_id,
                "file_name": self.image_path.name,
                "width": width,
                "height": height
            }],
            "annotations": [],
            "categories": [{"id": i, "name": name} for i, name in enumerate(self.classes)]
        }
        
        for i, box in enumerate(self.boxes):
            x, y, w, h = box.to_coco(width, height)
            coco_ann["annotations"].append({
                "id": i,
                "image_id": image_id,
                "category_id": box.class_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
        
        with open(self.output_path, 'w') as f:
            json.dump(coco_ann, f, indent=2)

def annotate_images(input_dir: Path, output_dir: Path, classes: List[str], format: str = 'yolo'):
    """Annotate images with bounding boxes
    
    Args:
        input_dir: Directory containing images to annotate
        output_dir: Directory to save annotations
        classes: List of class names
        format: Annotation format ('yolo' or 'coco')
    """
    # Create output directory structure for YOLO format
    if format == 'yolo':
        train_dir = output_dir / 'train'
        val_dir = output_dir / 'val'
        train_images_dir = train_dir / 'images'
        train_labels_dir = train_dir / 'labels'
        val_images_dir = val_dir / 'images'
        val_labels_dir = val_dir / 'labels'
        
        # Create all necessary directories
        for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
    if not image_files:
        raise ValueError(f"No images found in {input_dir}")
    
    # Sort files for consistent order
    image_files.sort()
    
    # Track current image index
    current_idx = 0
    
    while current_idx < len(image_files):
        image_path = image_files[current_idx]
        print(f"\nAnnotating image {current_idx + 1}/{len(image_files)}: {image_path.name}")
        
        # Determine output paths based on format
        if format == 'yolo':
            # Use train/val split based on index
            if current_idx < int(len(image_files) * 0.8):
                output_path = train_labels_dir / f"{image_path.stem}.txt"
                image_output_path = train_images_dir / image_path.name
            else:
                output_path = val_labels_dir / f"{image_path.stem}.txt"
                image_output_path = val_images_dir / image_path.name
        else:
            output_path = output_dir / f"{image_path.stem}.json"
            image_output_path = output_dir / image_path.name
        
        # Create annotator
        annotator = ImageAnnotator(image_path, classes, output_path, format, current_idx + 1, len(image_files))
        
        # Run annotation interface
        result = annotator.run()
        
        # Save annotations and copy image before navigation
        annotator.save_annotations()
        if format == 'yolo':
            shutil.copy2(image_path, image_output_path)
        
        if result == 'back':
            if current_idx > 0:
                # Go back one image
                current_idx -= 1
            else:
                print("\033[1;33mAlready at the first image.\033[0m")
        elif result == 'forward':
            if current_idx < len(image_files) - 1:
                # Go forward one image
                current_idx += 1
            else:
                print("\033[1;33mAlready at the last image.\033[0m")
        elif result == True:
            # Move to next image
            current_idx += 1
        else:  # result == False (quit)
            break
    
    cv2.destroyAllWindows()
    print("\nAnnotation completed!") 