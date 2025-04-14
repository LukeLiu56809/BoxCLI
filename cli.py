from pathlib import Path
import click
from annotate import annotate_images
from utils.split import split_dataset
import os
import sys

ASCII_ART = r"""
______           _____  _     _____ 
| ___ \         /  __ \| |   |_   _|
| |_/ / _____  _| /  \/| |     | |  
| ___ \/ _ \ \/ / |    | |     | |  
| |_/ / (_) >  <| \__/\| |_____| |_ 
\____/ \___/_/\_\\____/\_____/\___/ 
"""

def print_welcome():
    """Print welcome message with ASCII art"""
    print("\033[1;36m" + ASCII_ART + "\033[0m")
    print("\033[1;32mWelcome to BoxCLI - Your Friendly Command Line Image Annotation Tool!\033[0m")
    print("\n\033[1;33mMouse Controls:\033[0m")
    print("  • Left click: Draw box")
    print("  • Right click: Delete last box")
    print("  • Middle click: Change class")
    print("\n\033[1;33mKeyboard Controls:\033[0m")
    print("  • n: Next class")
    print("  • p: Previous class")
    print("  • d: Delete last box")
    print("  • q: Save and quit")
    print("  • z: Go back to previous image")
    print("  • x: Go to next image")
    print("\n\033[1;33mNavigation:\033[0m")
    print("  • You can go back to previous images to fix mistakes")
    print("  • Your annotations are automatically saved when you quit")
    print("  • Use 'z' and 'x' keys to navigate between images")
    print("\033[1;33m" + "=" * 60 + "\033[0m")

def get_input_dir():
    """Get input directory from user"""
    while True:
        input_dir = input("\n\033[1;34mWhere are your input images stored? (path): \033[0m")
        input_path = Path(input_dir)
        if not input_path.exists():
            print("\033[1;31mError: Directory does not exist. Please try again.\033[0m")
            continue
        
        image_files = [f for f in input_path.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        if not image_files:
            print("\033[1;31mError: No images found in the specified directory.\033[0m")
            continue
            
        print(f"\n\033[1;32mFound {len(image_files)} images in the directory.\033[0m")
        confirm = input("Is this correct? (yes/no): ").lower()
        if confirm == 'yes':
            return input_path

def get_output_dir():
    """Get output directory from user"""
    while True:
        output_dir = input("\n\033[1;34mWhere would you like to save the annotations? (path): \033[0m")
        output_path = Path(output_dir)
        
        if output_path.exists() and any(output_path.iterdir()):
            print("\033[1;33mWarning: Output directory is not empty.\033[0m")
            confirm = input("Do you want to continue? (yes/no): ").lower()
            if confirm != 'yes':
                continue
        
        return output_path

def get_classes():
    """Get classes file from user"""
    while True:
        classes_file = input("\n\033[1;34mWhere is your classes file located? (path): \033[0m")
        classes_path = Path(classes_file)
        
        if not classes_path.exists():
            print("\033[1;31mError: Classes file does not exist.\033[0m")
            continue
            
        with open(classes_path) as f:
            classes = [line.strip() for line in f if line.strip()]
            
        if not classes:
            print("\033[1;31mError: No classes found in the file.\033[0m")
            continue
            
        print(f"\n\033[1;32mFound {len(classes)} classes: {', '.join(classes)}\033[0m")
        confirm = input("Is this correct? (yes/no): ").lower()
        if confirm == 'yes':
            return classes_path, classes

def get_format() -> str:
    """Get output format from user"""
    while True:
        format = input("\n\033[1;34mChoose output format (yolo/coco): \033[0m").lower()
        if format in ['yolo', 'coco']:
            return format
        print("\033[1;31mInvalid format. Please choose 'yolo' or 'coco'.\033[0m")

def main():
    """Main entry point for the CLI"""
    print_welcome()
    
    # Get input directory
    input_dir = get_input_dir()
    
    # Get output directory
    output_dir = get_output_dir()
    
    # Get classes
    classes_path, classes = get_classes()
    
    # Get format
    format = get_format()
    
    # Annotate images
    annotate_images(input_dir, output_dir, classes, format)

if __name__ == '__main__':
    main() 