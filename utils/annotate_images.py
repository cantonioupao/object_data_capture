import cv2
import numpy as np
from pathlib import Path
import json
import tkinter as tk
from tkinter import filedialog
import os
import shutil
import yaml

class ImageAnnotator:
    def __init__(self, images_dir: str, output_dir: str):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.current_box = []
        self.boxes = []
        self.drawing = False
        self.image = None
        self.image_name = None
        
        # Create output directories
        (self.output_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Window setup
        cv2.namedWindow('Image Annotator')
        cv2.setMouseCallback('Image Annotator', self.mouse_callback)
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_box = [(x, y)]
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                img_copy = self.image.copy()
                cv2.rectangle(img_copy, self.current_box[0], (x, y), (0, 255, 0), 2)
                cv2.imshow('Image Annotator', img_copy)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.current_box.append((x, y))
            cv2.rectangle(self.image, self.current_box[0], self.current_box[1], (0, 255, 0), 2)
            self.boxes.append(self.current_box)
            cv2.imshow('Image Annotator', self.image)
            
    def convert_to_yolo(self, box, img_shape):
        """Convert (x1,y1,x2,y2) to YOLO format (center_x, center_y, width, height)"""
        x1, y1 = box[0]
        x2, y2 = box[1]
        
        # Ensure coordinates are in correct order
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Calculate center points and dimensions
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        # Normalize by image dimensions
        center_x /= img_shape[1]
        center_y /= img_shape[0]
        width /= img_shape[1]
        height /= img_shape[0]
        
        return [center_x, center_y, width, height]
    
    def annotate_images(self):
        image_files = list(self.images_dir.glob('*.jpg')) + \
                     list(self.images_dir.glob('*.jpeg')) + \
                     list(self.images_dir.glob('*.png'))
        
        for img_path in image_files:
            self.image_name = img_path.name
            self.image = cv2.imread(str(img_path))
            self.boxes = []
            
            print(f"\nAnnotating: {self.image_name}")
            print("Draw boxes with left mouse button")
            print("Press 's' to save and move to next image")
            print("Press 'r' to reset current image")
            print("Press 'q' to quit")
            
            while True:
                cv2.imshow('Image Annotator', self.image)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s'):  # Save and next
                    self.save_annotations(img_path)
                    break
                    
                elif key == ord('r'):  # Reset
                    self.image = cv2.imread(str(img_path))
                    self.boxes = []
                    cv2.imshow('Image Annotator', self.image)
                    
                elif key == ord('q'):  # Quit
                    cv2.destroyAllWindows()
                    return
                    
        cv2.destroyAllWindows()
        
    def save_annotations(self, img_path):
        """Save annotations in YOLO format"""
        # Copy image to output directory
        shutil.copy2(img_path, self.output_dir / 'images' / img_path.name)
        
        # Save YOLO format annotations
        label_path = self.output_dir / 'labels' / f"{img_path.stem}.txt"
        with open(label_path, 'w') as f:
            for box in self.boxes:
                # Convert to YOLO format
                yolo_box = self.convert_to_yolo(box, self.image.shape)
                # Write as: <class> <x> <y> <width> <height>
                f.write(f"0 {' '.join(map(str, yolo_box))}\n")
                
        print(f"Saved annotations for {img_path.name}")

def create_dataset_config(output_dir: str):
    """Create dataset configuration file"""
    config = {
        "path": str(output_dir),
        "train": "train",
        "val": "val",
        "nc": 1,  # number of classes
        "names": ["duck"]  # class names
    }
    
    with open(Path(output_dir) / 'dataset.yaml', 'w') as f:
        yaml.dump(config, f)

def main():
    # Select input directory
    root = tk.Tk()
    root.withdraw()
    images_dir = filedialog.askdirectory(title="Select Directory with Images")
    
    if not images_dir:
        print("No directory selected")
        return
        
    # Create output directory
    output_dir = Path(images_dir).parent / "annotated_dataset"
    
    # Initialize and run annotator
    annotator = ImageAnnotator(images_dir, output_dir)
    annotator.annotate_images()
    
    # Create dataset configuration
    create_dataset_config(output_dir)
    
    print("\nAnnotation completed!")
    print(f"Dataset created at: {output_dir}")

if __name__ == "__main__":
    main()