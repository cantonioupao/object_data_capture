import os
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

class TFLiteDatasetConverter:
    def __init__(self, yolo_dataset_dir: str, output_dir: str, val_split: float = 0.2):
        self.yolo_dir = Path(yolo_dataset_dir)
        self.output_dir = Path(output_dir)
        self.val_split = val_split
        
    def create_directory_structure(self):
        """Create TFLite compatible directory structure"""
        for split in ['train', 'val']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'annotations').mkdir(parents=True, exist_ok=True)
            
    def convert_yolo_to_xml(self, image_path: Path, label_path: Path) -> ET.Element:
        """Convert YOLO format annotation to XML format"""
        # Read image for dimensions
        image = cv2.imread(str(image_path))
        img_height, img_width = image.shape[:2]
        
        # Create XML structure
        annotation = ET.Element('annotation')
        
        # Add image information
        filename = ET.SubElement(annotation, 'filename')
        filename.text = image_path.name
        
        size = ET.SubElement(annotation, 'size')
        width = ET.SubElement(size, 'width')
        width.text = str(img_width)
        height = ET.SubElement(size, 'height')
        height.text = str(img_height)
        depth = ET.SubElement(size, 'depth')
        depth.text = '3'
        
        # Read YOLO annotations
        with open(label_path, 'r') as f:
            yolo_annotations = f.readlines()
            
        # Convert each YOLO box to XML format
        for yolo_box in yolo_annotations:
            parts = yolo_box.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            
            # Convert YOLO coordinates to pixel coordinates
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            
            # Calculate bbox coordinates
            xmin = int(x_center - width/2)
            ymin = int(y_center - height/2)
            xmax = int(x_center + width/2)
            ymax = int(y_center + height/2)
            
            # Create object element
            obj = ET.SubElement(annotation, 'object')
            
            name = ET.SubElement(obj, 'name')
            name.text = 'duck'  # or use class mapping if you have multiple classes
            
            bndbox = ET.SubElement(obj, 'bndbox')
            
            xml_xmin = ET.SubElement(bndbox, 'xmin')
            xml_xmin.text = str(max(0, xmin))
            xml_ymin = ET.SubElement(bndbox, 'ymin')
            xml_ymin.text = str(max(0, ymin))
            xml_xmax = ET.SubElement(bndbox, 'xmax')
            xml_xmax.text = str(min(img_width, xmax))
            xml_ymax = ET.SubElement(bndbox, 'ymax')
            xml_ymax.text = str(min(img_height, ymax))
            
        return annotation
    
    def convert_and_split_dataset(self):
        """Convert YOLO dataset to TFLite format and split into train/val"""
        # Get all image-label pairs
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list((self.yolo_dir / 'images').glob(f'*{ext}')))
            
        valid_pairs = []
        for img_path in image_files:
            label_path = self.yolo_dir / 'labels' / f"{img_path.stem}.txt"
            if label_path.exists():
                valid_pairs.append((img_path, label_path))
                
        if not valid_pairs:
            raise ValueError("No valid image-label pairs found")
            
        # Split into train/val
        train_pairs, val_pairs = train_test_split(
            valid_pairs,
            test_size=self.val_split,
            random_state=42
        )
        
        # Create directory structure
        self.create_directory_structure()
        
        # Process each split
        def process_split(pairs, split_name):
            for img_path, label_path in pairs:
                # Convert annotation to XML
                xml_annotation = self.convert_yolo_to_xml(img_path, label_path)
                
                # Save image
                shutil.copy2(
                    img_path,
                    self.output_dir / split_name / 'images' / img_path.name
                )
                
                # Save XML
                xml_path = self.output_dir / split_name / 'annotations' / f"{img_path.stem}.xml"
                tree = ET.ElementTree(xml_annotation)
                tree.write(xml_path, encoding='utf-8', xml_declaration=True)
                
        # Process both splits
        process_split(train_pairs, 'train')
        process_split(val_pairs, 'val')
        
        print(f"\nDataset conversion and split complete!")
        print(f"Training images: {len(train_pairs)}")
        print(f"Validation images: {len(val_pairs)}")
        
        return len(train_pairs), len(val_pairs)

def main():
    import argparse
    import tkinter as tk
    from tkinter import filedialog
    
    parser = argparse.ArgumentParser(description='Convert YOLO dataset to TFLite format and split')
    parser.add_argument('--val-split', type=float, default=0.2, 
                        help='Validation split ratio (default: 0.2)')
    
    args = parser.parse_args()
    
    # Create GUI file dialog
    root = tk.Tk()
    root.withdraw()
    
    # Get input directory
    yolo_dir = filedialog.askdirectory(
        title="Select YOLO dataset directory (contains images and labels folders)"
    )
    
    if not yolo_dir:
        print("No directory selected")
        return
        
    # Create output directory
    output_dir = Path(yolo_dir).parent / "tflite_dataset"
    
    # Initialize and run converter
    converter = TFLiteDatasetConverter(yolo_dir, output_dir, args.val_split)
    converter.convert_and_split_dataset()

if __name__ == "__main__":
    main()