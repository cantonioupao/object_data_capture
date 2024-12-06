import cv2
from object_detectors.simple_object_detection import SimpleObjectDetector
from utils.visualization import draw_bounding_box, visualize_image
from object_detectors.TFLite_mobilenet_detector import TFLiteDetector

# Initialize detector with model path
detector = TFLiteDetector(model_path='models/detect.tflite')
# test_detector by capturing one frame from the camera, showing the frame in a window and then show also the frame with the detection as a boudning box
def test_detector():
    detector = SimpleObjectDetector()
    cap = cv2.VideoCapture(0)
    
    ret, frame = cap.read()

    #Visualzie the frame
    visualize_image(frame)
    if ret:
        bbox = detector.detect_object(frame)['bounds']
        print(f"Detection result: {bbox}")
        # Draw bounding box on the frame
        frame_with_bbox = draw_bounding_box(frame, bbox)
        # Visualize captured frame and detection result
        cv2.imshow("Captured Frame", frame)
        cv2.imshow("Detection result", frame_with_bbox)
        cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_detector()


