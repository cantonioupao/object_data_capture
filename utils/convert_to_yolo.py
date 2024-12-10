import os
import xml.etree.ElementTree as ET
from pathlib import Path
import glob
import argparse
import shutil
from typing import Dict, List, Tuple
import cv2

class AnnotationConverter:
    def __init__(self, image_dir: str, annotation_dir: str, output_dir: str, class_file: str = None):
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.output_dir = Path(output_dir)
        self.classes = self.load_classes(class_file) if class_file else {}
        self.class_counter = len(self.classes)
        
        # Create output directories
        self.create_directories()
        
    def create_directories(self):
        """Create necessary output directories"""
        # Create main output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create images and labels directories
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'labels').mkdir(exist_ok=True)
        
    def load_classes(self, class_file: str) -> Dict[str, int]:
        """Load class names from file"""
        classes = {}
        if class_file and os.path.exists(class_file):
            with open(class_file, 'r') as f:
                for idx, line in enumerate(f.readlines()):
                    class_name = line.strip()
                    if class_name:
                        classes[class_name] = idx
        return classes
    
    def get_class_id(self, class_name: str) -> int:
        """Get class ID, create new if doesn't exist"""
        if class_name not in self.classes:
            self.classes[class_name] = self.class_counter
            self.class_counter += 1
        return self.classes[class_name]
    
    def convert_coordinates(self, size: Tuple[int, int], box: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
        """Convert VOC bbox coordinates to YOLO format"""
        dw = 1.0 / size[0]
        dh = 1.0 / size[1]
        
        # VOC format: xmin, ymin, xmax, ymax
        x = (box[0] + box[2]) / 2.0
        y = (box[1] + box[3]) / 2.0
        w = box[2] - box[0]
        h = box[3] - box[1]
        
        # Normalize
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        
        return (x, y, w, h)
    
    def process_xml(self, xml_file: Path, image_file: Path) -> List[str]:
        """Process single XML file and return YOLO format annotations"""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get image size
        img = cv2.imread(str(image_file))
        if img is None:
            raise ValueError(f"Could not read image: {image_file}")
        size = img.shape[1], img.shape[0]  # width, height
        
        yolo_annotations = []
        
        # Process each object in the XML
        for obj in root.findall('.//object'):
            # Get class name and convert to class id
            class_name = obj.find('name').text
            class_id = self.get_class_id(class_name)
            
            # Get bounding box coordinates
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Convert to YOLO format
            x, y, w, h = self.convert_coordinates(size, (xmin, ymin, xmax, ymax))
            
            # Create YOLO format line
            yolo_annotations.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        
        return yolo_annotations
    
    def convert_all(self):
        """Convert all annotations in directory"""
        # Get all image files
        image_files = list(self.image_dir.glob('*.jpg')) + list(self.image_dir.glob('*.jpeg')) + list(self.image_dir.glob('*.png'))
        
        print(f"Found {len(image_files)} images")
        
        for image_file in image_files:
            try:
                # Find corresponding XML file
                xml_file = self.annotation_dir / f"{image_file.stem}.xml"
                
                if not xml_file.exists():
                    print(f"Warning: No annotation file for {image_file}")
                    continue
                
                # Process XML and get YOLO annotations
                yolo_annotations = self.process_xml(xml_file, image_file)
                
                # Save YOLO format annotations
                output_label_file = self.output_dir / 'labels' / f"{image_file.stem}.txt"
                with open(output_label_file, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
                # Copy image to output directory
                shutil.copy2(image_file, self.output_dir / 'images' / image_file.name)
                
                print(f"Processed {image_file.name}")
                
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
        
        # Save class mapping
        with open(self.output_dir / 'classes.txt', 'w') as f:
            for class_name, class_id in sorted(self.classes.items(), key=lambda x: x[1]):
                f.write(f"{class_name}\n")
        
        print(f"\nConversion completed!")
        print(f"Total classes: {len(self.classes)}")
        print(f"Class mapping saved to {self.output_dir}/classes.txt")

def main():
    parser = argparse.ArgumentParser(description='Convert XML annotations to YOLO format')
    parser.add_argument('--image-dir', required=True, help='Directory containing images')
    parser.add_argument('--annotation-dir', required=True, help='Directory containing XML annotations')
    parser.add_argument('--output-dir', required=True, help='Output directory for YOLO format dataset')
    parser.add_argument('--class-file', help='Text file containing class names (optional)')
    
    args = parser.parse_args()
    
    converter = AnnotationConverter(
        args.image_dir,
        args.annotation_dir,
        args.output_dir,
        args.class_file
    )
    
    converter.convert_all()



if __name__ == '__main__':
    main()


# Add acomment for correct usage
# Usage:
# Add comment with many lines
# python convert_to_yolo.py --image-dir path/to/images
# --annotation-dir path/to/annotations 
# --output-dir path/to/output 
# --class-file path/to/classes.txt
