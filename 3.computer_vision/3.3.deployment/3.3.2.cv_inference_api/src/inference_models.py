import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.schemas import (
    ObjectDetectionPrediction, 
    ClassificationPrediction, 
    PoseDetectionPrediction,
    EmbeddingPrediction
)



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
    
class ImageEmbedder:
    def __init__(self, model_path: str = "embeddings_model.tflite", l2_normalize: bool = True, quantize:bool = True):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ImageEmbedderOptions(
                    base_options=base_options, 
                    l2_normalize=l2_normalize, 
                    quantize=quantize
        )
        self.model = vision.ImageEmbedder.create_from_options(options)

    def predict(self, img_array: np.ndarray) -> EmbeddingPrediction:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
        results = self.model.embed(image)
        embedding = results.embeddings[0].embedding.tolist()
        return EmbeddingPrediction(
            n_dim=len(embedding),
            embedding=embedding
        )