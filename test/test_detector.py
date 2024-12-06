import cv2
from object_detectors.simple_object_detection import SimpleObjectDetector

def test_detector():
    detector = SimpleObjectDetector()
    cap = cv2.VideoCapture(0)
    
    ret, frame = cap.read()
    if ret:
        result = detector.detect_object(frame)
        print(f"Detection result: {result}")
    
    cap.release()


if __name__ == '__main__':
    test_detector()    