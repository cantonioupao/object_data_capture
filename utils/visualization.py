import cv2
from typing import List

def draw_bounding_box(frame, bbox):
    x, y, w, h = bbox
    return cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

def visualize_image(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_text(frame, text: List[str]):
    # Visualize text on image frame using OpenCV
    for i, txt in enumerate(text):
        cv2.putText(frame, txt, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame
