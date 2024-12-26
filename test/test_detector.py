import cv2
from object_detectors.simple_object_detection import SimpleObjectDetector
from utils.visualization import draw_bounding_box, visualize_image, visualize_text, visualize_fps
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
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Get detection results
            detection = detector.detect_object(frame)
            
            # Add FPS visualization first (so it's always on top)
            realtime_fps = detector.get_fps()
            inference_fps = detector.inference_fps
            frame = visualize_fps(frame, realtime_fps)

            
            if detection:
                # Draw bounding box with integrated label
                label = f"{detection['class']}: {detection['score']:.2f}"
                frame = draw_bounding_box(frame, detection['bounds'], label)
                
                # Add additional information
                info_text = [
                    #f"Inference Time: {detection['inference_time']*1000:.1f}ms",
                    f"Inference FPS: {inference_fps:.1f}",
                    #f"Object Size: {detection['bounds'][2]}x{detection['bounds'][3]}px"
                ]
                frame = visualize_text(frame, info_text)
            else:
                # Show "No Detection" message
                frame = visualize_text(frame, ["No objects detected"])
            
            cv2.imshow('Object Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error in camera feed: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    test_detector_on_camera_feed()


