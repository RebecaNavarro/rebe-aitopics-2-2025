import numpy as np
import cv2
from ultralytics import YOLO
from schemas import ObjectDetectionPrediction, ClassificationPrediction, PoseDetectionPrediction

class ObjectDetector:
    def __init__(self, model_name: str = "yolo11n.pt", conf_threshold: float = 0.5):
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold

    def predict(self, img_array: np.ndarray) -> ObjectDetectionPrediction:
        results = self.model(img_array, conf=self.conf_threshold)[0]
        labels = [results.names[idx] for idx in results.boxes.cls.tolist()]
        boxes = results.boxes.xyxy.int().tolist()
        scores = results.boxes.conf.tolist()

        return ObjectDetectionPrediction(
            n_detections=len(boxes),
            boxes=boxes,
            labels=labels,
            scores=scores
        )
    
    def predict_with_image(self, img_array: np.ndarray) -> np.ndarray:
        results = self.model(img_array, conf=self.conf_threshold)[0]
        annotated_img = results.plot()  
        return annotated_img
    
class Classificator:
    def __init__(self, model_name: str = "yolo11n-cls.pt", conf_threshold: float = 0.5):
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold

    def predict(self, img_array: np.ndarray) -> ClassificationPrediction:
        results = self.model(img_array, conf=self.conf_threshold)[0]
        label = results.names[results.probs.top1]
        score = results.probs.top1conf

        return ClassificationPrediction(
            label=label,
            score=score
        )
    
    def predict_with_image(self, img_array: np.ndarray) -> np.ndarray:
        results = self.model(img_array, conf=self.conf_threshold)[0]
        label = results.names[results.probs.top1]
        score = results.probs.top1conf
        
        annotated_img = img_array.copy()
        
        text = f"{label}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        cv2.rectangle(annotated_img, (10, 10), (20 + text_width, 40 + text_height), (0, 0, 0), -1)
        
        cv2.putText(annotated_img, text, (20, 40), font, font_scale, (255, 255, 255), thickness)
        
        return annotated_img
   

class PoseDetector:
    def __init__(self, model_name: str = "yolo11n-pose.pt"):
        self.model = YOLO(model_name)

    def predict(self, img_array: np.ndarray) -> ObjectDetectionPrediction:
        results = self.model([img_array])[0]
        keypoints = results.keypoints.xy.int().tolist()
        visibility = results.keypoints.data[:, :, -1].tolist()

        return PoseDetectionPrediction(
            n_detections=len(keypoints),
            keypoints=keypoints,
            visibility=visibility
        )
    
    def predict_with_image(self, img_array: np.ndarray) -> np.ndarray:
        results = self.model(img_array)[0]
        annotated_img = results.plot()  
        return annotated_img