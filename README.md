# BoxCLI

A simple command-line tool for drawing bounding boxes on images. Supports YOLO and COCO formats.

## Quick Start

1. Install:
```bash
pip install -e .
```

2. Run:
```bash
boxcli
```

3. Follow the prompts:
   - Enter your input images directory
   - Choose where to save annotations
   - Select your classes file
   - Pick format (yolo/coco)

## Controls

### Mouse
- Left click: Draw box
- Right click: Delete last box
- Middle click: Change class

### Keyboard
- n: Next class
- p: Previous class
- d: Delete last box
- z: Go back to previous image
- x: Go to next image
- q: Save and quit

## Classes File

Create a text file with one class per line:
```
person
car
dog
cat
```

## Output Formats

### YOLO
- Creates train/val split (80/20)
- Saves annotations as .txt files
- Normalized coordinates (0-1)

### COCO
- Single JSON file
- Standard COCO format
- Includes image metadata

## Requirements
- Python 3.6+
- OpenCV
- Click
- tqdm 