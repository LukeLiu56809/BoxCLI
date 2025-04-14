from pathlib import Path
from typing import List, Dict, Any
import json
import cv2
from annotator import BoundingBox

class COCOConverter:
    def __init__(self):
        self.coco_data = {
            "info": {},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        self.image_id = 1
        self.annotation_id = 1

    def add_image(self, image_path: Path, image_id: int = None):
        """Add image information to COCO dataset"""
        if image_id is None:
            image_id = self.image_id
            self.image_id += 1

        img = cv2.imread(str(image_path))
        height, width = img.shape[:2]

        image_info = {
            "id": image_id,
            "file_name": image_path.name,
            "width": width,
            "height": height
        }
        self.coco_data["images"].append(image_info)
        return image_id

    def add_category(self, category_name: str, category_id: int = None):
        """Add category information to COCO dataset"""
        if category_id is None:
            category_id = len(self.coco_data["categories"]) + 1

        category = {
            "id": category_id,
            "name": category_name
        }
        self.coco_data["categories"].append(category)
        return category_id

    def add_annotation(self, image_id: int, category_id: int, bbox: BoundingBox, annotation_id: int = None):
        """Add annotation to COCO dataset"""
        if annotation_id is None:
            annotation_id = self.annotation_id
            self.annotation_id += 1

        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1],
            "area": (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1),
            "iscrowd": 0
        }
        self.coco_data["annotations"].append(annotation)

    def save(self, output_path: Path):
        """Save COCO dataset to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.coco_data, f, indent=2)

    @staticmethod
    def load_annotations(file_path: Path, classes: List[str]) -> Dict[str, List[BoundingBox]]:
        """Load annotations from COCO format file"""
        with open(file_path, 'r') as f:
            coco_data = json.load(f)

        annotations = {}
        category_id_to_class_id = {cat["id"]: idx for idx, cat in enumerate(coco_data["categories"])}

        for img in coco_data["images"]:
            image_annotations = []
            for ann in coco_data["annotations"]:
                if ann["image_id"] == img["id"]:
                    bbox = ann["bbox"]
                    class_id = category_id_to_class_id[ann["category_id"]]
                    image_annotations.append(BoundingBox(
                        int(bbox[0]),
                        int(bbox[1]),
                        int(bbox[0] + bbox[2]),
                        int(bbox[1] + bbox[3]),
                        class_id
                    ))
            annotations[img["file_name"]] = image_annotations

        return annotations 