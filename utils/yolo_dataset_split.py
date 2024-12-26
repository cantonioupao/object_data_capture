import os
import shutil
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
import yaml
import argparse
import tkinter as tk
from tkinter import filedialog

class YOLODatasetSplitter:
    def __init__(self, dataset_dir: str, output_dir: str, val_split: float = 0.2):
        """
        Initialize dataset splitter for YOLO format annotations.
        
        Args:
            dataset_dir: Directory containing images and labels
            output_dir: Directory to save split dataset
            val_split: Fraction of data to use for validation
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.val_split = val_split
        
    def create_directory_structure(self):
        """Create the necessary directory structure for YOLO format"""
        for split in ['train', 'val']:
            # Create images and labels directories for each split
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
            
    def get_image_label_pairs(self):
        """Get pairs of images and their corresponding label files"""
        images_dir = self.dataset_dir / 'images'
        labels_dir = self.dataset_dir / 'labels'
        
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(images_dir.glob(f'*{ext}')))
            
        # Create pairs of image and label files
        pairs = []
        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                pairs.append((img_path, label_path))
            else:
                print(f"Warning: No label file found for {img_path.name}")
                
        return pairs
        
    def split_dataset(self):
        """Split the dataset into training and validation sets"""
        # Get file pairs
        pairs = self.get_image_label_pairs()
        
        if not pairs:
            raise ValueError("No valid image-label pairs found")
            
        # Split pairs into train and validation
        train_pairs, val_pairs = train_test_split(
            pairs,
            test_size=self.val_split,
            random_state=42
        )
        
        # Create directories
        self.create_directory_structure()
        
        # Copy files to respective directories
        def copy_pairs(pairs, split):
            for img_path, label_path in pairs:
                # Copy image
                shutil.copy2(
                    img_path,
                    self.output_dir / split / 'images' / img_path.name
                )
                # Copy label
                shutil.copy2(
                    label_path,
                    self.output_dir / split / 'labels' / label_path.name
                )
                
        # Copy files
        copy_pairs(train_pairs, 'train')
        copy_pairs(val_pairs, 'val')
        
        # Create dataset YAML configuration
        self.create_dataset_yaml(len(train_pairs), len(val_pairs))
        
        print(f"\nDataset split complete!")
        print(f"Training images: {len(train_pairs)}")
        print(f"Validation images: {len(val_pairs)}")
        
        return len(train_pairs), len(val_pairs)
        
    def create_dataset_yaml(self, num_train: int, num_val: int):
        """Create YOLO format dataset configuration file"""
        config = {
            'path': str(self.output_dir),  # Dataset root directory
            'train': str(Path('train/images')),  # Train images relative to path
            'val': str(Path('val/images')),      # Val images relative to path
            'nc': 1,  # Number of classes
            'names': ['duck'],  # Class names
            
            # Additional information
            'stats': {
                'total_images': num_train + num_val,
                'training_images': num_train,
                'validation_images': num_val,
                'split_ratio': {
                    'train': 1 - self.val_split,
                    'val': self.val_split
                }
            }
        }
        
        with open(self.output_dir / 'dataset.yaml', 'w') as f:
            yaml.dump(config, f, sort_keys=False, default_flow_style=False)

def main():    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Split YOLO format dataset into training and validation sets')
    parser.add_argument('--val-split', type=float, default=0.2, 
                        help='Validation split ratio (default: 0.2)')
    
    args = parser.parse_args()
    
    # Create GUI file dialog
    root = tk.Tk()
    root.withdraw()
    
    # Get input directory (should be the annotated_dataset directory)
    dataset_dir = filedialog.askdirectory(
        title="Select annotated dataset directory (contains images and labels folders)"
    )
    
    if not dataset_dir:
        print("No directory selected")
        return
        
    # Create output directory
    output_dir = Path(dataset_dir).parent / "split_dataset"
    
    # Initialize and run splitter
    splitter = YOLODatasetSplitter(dataset_dir, output_dir, args.val_split)
    splitter.split_dataset()

if __name__ == "__main__":
    main()