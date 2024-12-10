import tensorflow as tf
from typing import Dict, List
import numpy as np

class ObjectDetectionMetrics:
    """Metrics calculator for object detection"""
    
    def __init__(self):
        self.iou_threshold = 0.5
        self.reset_states()
        
    def reset_states(self):
        """Reset metric states"""
        self.total_instances = 0
        self.true_positives = []
        self.false_positives = []
        self.scores = []
        
    def update_state(self, true_classes: tf.Tensor, true_boxes: tf.Tensor,
                    pred_classes: tf.Tensor, pred_boxes: tf.Tensor) -> Dict:
        """Update metric states with batch results"""
        batch_size = tf.shape(true_classes)[0]
        
        for i in range(batch_size):
            matched_indices = self._match_boxes(
                true_boxes[i],
                pred_boxes[i],
                self.iou_threshold
            )
            
            # Update metrics based on matches
            self._update_batch_metrics(
                true_classes[i],
                pred_classes[i],
                matched_indices
            )
            
        return self.result()
        
    def _match_boxes(self, true_boxes: tf.Tensor, pred_boxes: tf.Tensor,
                    iou_threshold: float) -> tf.Tensor:
        """Match predicted boxes to ground truth boxes based on IoU"""
        iou_matrix = self._calculate_iou(true_boxes, pred_boxes)
        matched_indices = tf.zeros_like(true_boxes[..., 0], dtype=tf.int32)
        
        # Greedy matching
        while tf.reduce_max(iou_matrix) >= iou_threshold:
            max_idx = tf.argmax(tf.reshape(iou_matrix, [-1]))
            true_idx = max_idx // tf.shape(pred_boxes)[0]
            pred_idx = max_idx % tf.shape(pred_boxes)[0]
            
            matched_indices = tf.tensor_scatter_nd_update(
                matched_indices,
                [[true_idx]],
                [pred_idx]
            )
            
            # Zero out matched entries
            iou_matrix = tf.tensor_scatter_nd_update(
                iou_matrix,
                [[true_idx, pred_idx]],
                [0.0]
            )
            
        return matched_indices
        
    def _calculate_iou(self, boxes1: tf.Tensor, boxes2: tf.Tensor) -> tf.Tensor:
        """Calculate IoU between two sets of boxes"""
        # Calculate intersection areas
        x1 = tf.maximum(boxes1[..., 0][:, None], boxes2[..., 0])
        y1 = tf.maximum(boxes1[..., 1][:, None], boxes2[..., 1])
        x2 = tf.minimum(boxes1[..., 2][:, None], boxes2[..., 2])
        y2 = tf.minimum(boxes1[..., 3][:, None], boxes2[..., 3])
        
        intersection = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)
        
        # Calculate areas
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        
        union = boxes1_area[:, None] + boxes2_area - intersection
        
        return intersection / union
        
    def _update_batch_metrics(self, true_classes: tf.Tensor, pred_classes: tf.Tensor,
                           matched_indices: tf.Tensor):
        """Update metrics based on matched predictions"""
        self.total_instances += tf.shape(true_classes)[0]
        
        # Calculate true positives and false positives
        correct_class = tf.equal(
            true_classes,
            tf.gather(pred_classes, matched_indices)
        )
        
        self.true_positives.append(tf.cast(correct_class, tf.float32))
        self.false_positives.append(
            tf.cast(tf.logical_not(correct_class), tf.float32)
        )
        
    def result(self) -> Dict:
        """Calculate final metrics"""
        if not self.true_positives:
            return {
                'mAP': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }
            
        # Calculate precision and recall
        tp = tf.concat(self.true_positives, axis=0)
        fp = tf.concat(self.false_positives, axis=0)
        
        precision = tf.reduce_sum(tp) / (tf.reduce_sum(tp) + tf.reduce_sum(fp) + 1e-7)
        recall = tf.reduce_sum(tp) / (self.total_instances + 1e-7)
        
        return {
            'mAP': self._calculate_map(),
            'precision': precision.numpy(),
            'recall': recall.numpy()
        }
        
    def _calculate_map(self) -> float:
        """Calculate mean Average Precision"""
        # Implementation of mAP calculation
        # This is a simplified version
        precisions = tf.reduce_sum(self.true_positives) / (
            tf.reduce_sum(self.true_positives) + tf.reduce_sum(self.false_positives)
        )
        return float(precisions.numpy())
