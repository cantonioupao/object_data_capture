import cv2
from object_detectors.simple_object_detection import SimpleObjectDetector
from utils.visualization import draw_bounding_box, visualize_image, visualize_text
from object_detectors.TFLite_mobilenet_detector import TFLiteDetector


# test_detector by capturing one frame from the camera, showing the frame in a window and then show also the frame with the detection as a boudning box
def test_detector():
    #detector = SimpleObjectDetector()
    detector = TFLiteDetector(model_path='models/mobilenet_ssd_tflite/detect.tflite', 
                              model_labels_path='models/mobilenet_ssd_tflite/labelmap.txt',
                              confidence_threshold=0.4)
    cap = cv2.VideoCapture(0)
    
    ret, frame = cap.read()

    #Visualzie the frame
    visualize_image(frame)
    if ret:
        detection_results = detector.detect_object(frame)
        bbox = detection_results['bounds']
        print(f"Detection result: {bbox}")
        # Draw bounding box on the frame
        frame_with_bbox = draw_bounding_box(frame.copy(), bbox)
        # Visualize detection result
        cv2.imshow("Detection result", frame_with_bbox)
        cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

def test_detector_on_camera_feed():
    detector = TFLiteDetector(model_path='models/mobilenet_ssd_tflite/detect.tflite', 
                              model_labels_path='models/mobilenet_ssd_tflite/labelmap.txt',
                              confidence_threshold=0.4)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            detection_results = detector.detect_object(frame)
            if detection_results is None:
                continue
            bbox = detection_results['bounds']
            print(f"Detection result: {bbox}")
            # Draw bounding box on the frame
            frame_with_bbox = draw_bounding_box(frame.copy(), bbox)
            class_name = detection_results['class']
            print(f"Class: {class_name}")
            frame_with_bbox_and_text = visualize_text(frame_with_bbox.copy(), [class_name])
            # Visualize detection result
            cv2.imshow("Detection result", frame_with_bbox_and_text)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == '__main__':
    test_detector_on_camera_feed()


