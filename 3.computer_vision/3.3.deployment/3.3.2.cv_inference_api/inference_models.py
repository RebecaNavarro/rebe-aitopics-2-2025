import numpy as np
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