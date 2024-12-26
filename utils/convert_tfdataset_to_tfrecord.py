"""
TFRecord Dataset Converter for Object Detection

This script converts a dataset organized in the following structure:
train/
    ├── annotations/  (XML files)
    └── images/       (JPEG files)
val/
    ├── annotations/  (XML files)
    └── images/       (JPEG files)

Into the TFRecord format required by TensorFlow Object Detection API.

The conversion process:
1. Reads JPEG images and their corresponding XML annotations
2. Parses XML files to extract bounding box coordinates and class labels
3. Normalizes bounding box coordinates to [0,1] range
4. Creates TFRecord examples containing:
   - Encoded JPEG image data
   - Image dimensions (height, width)
   - Normalized bounding box coordinates (xmin, ymin, xmax, ymax)
   - Class labels ('duck' in this case)
5. Writes the examples to TFRecord files (train.record and val.record)
6. Creates a label map file (label_map.pbtxt) mapping class names to IDs

Usage:
1. Command line: python convert_tfdataset_to_tfrecord.py --input_dir path/to/dataset
2. Interactive: python convert_tfdataset_to_tfrecord.py --interactive
"""

import os
import xml.etree.ElementTree as ET
from object_detection.utils import dataset_util
import tensorflow as tf
from pathlib import Path
import glob
import argparse
import tkinter as tk
from tkinter import filedialog
import sys

def select_folder():
    """Open folder selection dialog"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    folder_path = filedialog.askdirectory(
        title='Select Dataset Root Directory',
        mustexist=True
    )
    
    return folder_path if folder_path else None

def verify_dataset_structure(dataset_dir):
    """Verify the dataset has the correct directory structure"""
    required_dirs = [
        os.path.join(dataset_dir, 'train', 'images'),
        os.path.join(dataset_dir, 'train', 'annotations'),
        os.path.join(dataset_dir, 'val', 'images'),
        os.path.join(dataset_dir, 'val', 'annotations')
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"Error: Directory not found: {directory}")
            print("Required structure:")
            print("dataset_root/")
            print("    ├── train/")
            print("    │   ├── annotations/")
            print("    │   └── images/")
            print("    └── val/")
            print("        ├── annotations/")
            print("        └── images/")
            return False
    
    return True

def parse_xml_annotation(xml_path):
    """Parse XML annotation file"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get bounding box
    boxes = []
    for obj in root.findall('.//object'):
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
    
    return boxes

def create_tf_example(image_path, xml_path):
    """Create TF Example from image and annotation"""
    # Read image
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image = fid.read()
    
    # Get image dimensions
    image = tf.image.decode_jpeg(encoded_image)
    height, width = image.shape[:2]
    
    # Parse annotation
    boxes = parse_xml_annotation(xml_path)
    
    # Normalize box coordinates
    xmins = [box[0]/width for box in boxes]
    xmaxs = [box[2]/width for box in boxes]
    ymins = [box[1]/height for box in boxes]
    ymaxs = [box[3]/height for box in boxes]
    
    # Create feature dictionary
    feature = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(os.path.basename(image_path).encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(os.path.basename(image_path).encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(['duck'.encode('utf8')] * len(boxes)),
        'image/object/class/label': dataset_util.int64_list_feature([1] * len(boxes))
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))

def convert_dataset(dataset_dir, output_path):
    """Convert entire dataset to TFRecord"""
    writer = tf.io.TFRecordWriter(output_path)
    
    # Get all image and annotation files
    image_dir = os.path.join(dataset_dir, 'images')
    annot_dir = os.path.join(dataset_dir, 'annotations')
    
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    total_images = len(image_paths)
    
    print(f"Found {total_images} images in {image_dir}")
    
    for i, image_path in enumerate(image_paths, 1):
        # Get corresponding annotation file
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        xml_path = os.path.join(annot_dir, f"{base_name}.xml")
        
        if os.path.exists(xml_path):
            tf_example = create_tf_example(image_path, xml_path)
            writer.write(tf_example.SerializeToString())
            
            # Show progress
            print(f"\rProcessing: {i}/{total_images} ({(i/total_images)*100:.1f}%)", end='')
        else:
            print(f"\nWarning: No annotation found for {base_name}")
    
    print("\nDone!")
    writer.close()

def create_label_map(output_path):
    """Create label map file"""
    with open(output_path, 'w') as f:
        f.write("""item {
    id: 1
    name: 'duck'
}""")

def main():
    parser = argparse.ArgumentParser(description='Convert dataset to TFRecord format')
    parser.add_argument('--input_dir', type=str, help='Path to dataset root directory')
    parser.add_argument('--interactive', action='store_true', help='Use interactive folder selection')
    parser.add_argument('--output_dir', type=str, default='tfrecord_data', help='Output directory for TFRecords')
    
    args = parser.parse_args()
    
    # Get input directory
    if args.interactive:
        print("Please select the dataset root directory...")
        dataset_dir = select_folder()
        if not dataset_dir:
            print("No folder selected. Exiting...")
            sys.exit(1)
    else:
        dataset_dir = args.input_dir
        if not dataset_dir:
            parser.error("Please provide --input_dir or use --interactive")
    
    # Verify dataset structure
    if not verify_dataset_structure(dataset_dir):
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\nConverting training set...")
    convert_dataset(
        os.path.join(dataset_dir, 'train'),
        os.path.join(args.output_dir, 'train.record')
    )
    
    print("\nConverting validation set...")
    convert_dataset(
        os.path.join(dataset_dir, 'val'),
        os.path.join(args.output_dir, 'val.record')
    )
    
    # Create label map
    label_map_path = os.path.join(args.output_dir, 'label_map.pbtxt')
    create_label_map(label_map_path)
    print(f"\nCreated label map at {label_map_path}")
    
    print(f"\nConversion complete! Output files are in: {args.output_dir}")

if __name__ == "__main__":
    main()