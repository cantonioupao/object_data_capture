import cv2
from typing import List

def draw_bounding_box(frame, bbox):
    """Draw bounding box with label"""
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

def visualize_image(image):
    """Display image until key press"""
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_detection(frame, detection):
    """Draw detection box, class and confidence"""
    if not detection:
        return frame
    x, y, w, h = detection['bounds']
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    label = f"{detection['class']}: {detection['score']:.2f}"
    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return 


def draw_bounding_box(frame, bbox, label=None):
    """Draw bounding box and optional label"""
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    if label:
        # Add background for text visibility
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x, y - 25), (x + label_size[0] + 10, y), (0, 255, 0), -1)
        cv2.putText(frame, label, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame

def visualize_text(frame, text: List[str]):
    """Add text list to frame with dark background for visibility"""
    for i, txt in enumerate(text):
        y_pos = 60 + 30 * i  # Start below FPS counter
        txt_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        # Draw background
        cv2.rectangle(frame, (10, y_pos - 20), (txt_size[0] + 20, y_pos + 5), (0, 0, 0), -1)
        # Draw text
        cv2.putText(frame, txt, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

def visualize_fps(frame, fps):
    """Add FPS counter with background"""
    fps_text = f"FPS: {fps:.1f}"
    txt_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    # Draw background
    cv2.rectangle(frame, (10, 10), (txt_size[0] + 20, 40), (0, 0, 0), -1)
    # Draw text
    cv2.putText(frame, fps_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame