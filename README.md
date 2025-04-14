# BoxCLI

A simple command-line tool for drawing bounding boxes on images. Supports YOLO and COCO formats.

## Quick Start

1. Create and activate virtual environment:
```bash
# Create venv
python -m venv boxcli_env

# Activate venv
# On Windows:
boxcli_env\Scripts\activate
# On macOS/Linux:
source boxcli_env/bin/activate
```

2. Install:
```bash
pip install -e .
```

3. Run:
```bash
boxcli
```

4. Follow the prompts:
   - Enter your input images directory
   - Choose where to save annotations
   - Select your classes file (see example below)
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

Create a text file named `classes.txt` with one class per line. Example:
```
person
car
truck
bicycle
motorcycle
bus
traffic light
stop sign
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

## License

MIT License

Copyright (c) 2024 BoxCLI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 