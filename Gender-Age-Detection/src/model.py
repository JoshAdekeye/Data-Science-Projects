import cv2
import numpy as np
import os

class GenderAgeDetector:
    def __init__(self, model_path=None):
        """Initialize the detector with model paths."""
        if model_path is None:
            # Get the absolute path to the project root directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(os.path.dirname(current_dir), 'models')
        
        self.model_path = model_path
        self.face_proto = os.path.join(model_path, "opencv_face_detector.pbtxt")
        self.face_model = os.path.join(model_path, "opencv_face_detector_uint8.pb")
        self.age_proto = os.path.join(model_path, "age_deploy.prototxt")
        self.age_model = os.path.join(model_path, "age_net.caffemodel")
        self.gender_proto = os.path.join(model_path, "gender_deploy.prototxt")
        self.gender_model = os.path.join(model_path, "gender_net.caffemodel")

        # Verify model files exist
        for model_file in [self.face_proto, self.face_model, self.age_proto, 
                          self.age_model, self.gender_proto, self.gender_model]:
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Model file not found: {model_file}")

        # Model parameters
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.gender_list = ['Male', 'Female']

        # Load networks
        self.face_net = cv2.dnn.readNet(self.face_model, self.face_proto)
        self.age_net = cv2.dnn.readNet(self.age_model, self.age_proto)
        self.gender_net = cv2.dnn.readNet(self.gender_model, self.gender_proto)

    def highlight_face(self, frame, conf_threshold=0.7):
        """Detect and highlight faces in the frame."""
        frame_opencv_dnn = frame.copy()
        frame_height = frame_opencv_dnn.shape[0]
        frame_width = frame_opencv_dnn.shape[1]
        
        blob = cv2.dnn.blobFromImage(
            frame_opencv_dnn, 1.0, (300, 300), 
            [104, 117, 123], True, False
        )

        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        face_boxes = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)
                face_boxes.append([x1, y1, x2, y2])
                cv2.rectangle(
                    frame_opencv_dnn, (x1, y1), (x2, y2),
                    (0, 255, 0), int(round(frame_height/150)), 8
                )

        return frame_opencv_dnn, face_boxes

    def predict_age_gender(self, frame, face_box, padding=20):
        """Predict age and gender for a detected face."""
        face = frame[
            max(0, face_box[1]-padding):
            min(face_box[3]+padding, frame.shape[0]-1),
            max(0, face_box[0]-padding):
            min(face_box[2]+padding, frame.shape[1]-1)
        ]

        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227),
            self.MODEL_MEAN_VALUES, swapRB=False
        )

        # Gender prediction
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = self.gender_list[gender_preds[0].argmax()]

        # Age prediction
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age = self.age_list[age_preds[0].argmax()]

        return gender, age

    def process_image(self, image):
        """Process a single image and return results.
        
        Args:
            image: Can be either a string (path to image file) or a numpy array
        """
        if isinstance(image, str):
            frame = cv2.imread(image)
            if frame is None:
                raise ValueError(f"Unable to load image from path: {image}")
        elif isinstance(image, np.ndarray):
            frame = image
        else:
            raise ValueError("Image must be either a file path (string) or a numpy array")

        result_img, face_boxes = self.highlight_face(frame)
        results = []

        for face_box in face_boxes:
            gender, age = self.predict_age_gender(frame, face_box)
            results.append({
                'gender': gender,
                'age': age.strip('()'),
                'box': face_box
            })
            
            # Add text to image
            cv2.putText(
                result_img, 
                f'{gender}, {age}', 
                (face_box[0], face_box[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2,
                cv2.LINE_AA
            )

        return result_img, results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to input image')
    args = parser.parse_args()

    detector = GenderAgeDetector()
    result_image, predictions = detector.process_image(args.image)
    
    for pred in predictions:
        print(f"Found face with Gender: {pred['gender']}, Age: {pred['age']} years")
    
    cv2.imshow("Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 