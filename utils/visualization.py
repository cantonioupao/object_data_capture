import cv2

def draw_bounding_box(frame, bbox):
    x, y, w, h = bbox
    return cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

def visualize_image(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()