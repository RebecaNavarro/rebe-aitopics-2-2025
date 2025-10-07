from pydantic import BaseModel
from enum import Enum

POSE_POINTS = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]

class PredictionType(str, Enum):
    object_detection = "OD"
    segmentation = "SEG"
    classification = "CLS"
    pose_detection = "POSE"
    keypoint_detection = "KP"
    embedding = "EMB"

class ImageMetadata(BaseModel):
    file_name: str
    height: int
    width: int
    channels: int


class ObjectDetectionPrediction(BaseModel):
    n_detections: int
    boxes: list[list[int]]
    labels: list[str]
    scores: list[float]


class ClassificationPrediction(BaseModel):
    label: str
    score: float


class PoseDetectionPrediction(BaseModel):
    n_detections: int
    keypoints: list[list[list[int]]]
    visibility: list[list[float]]
    point_labels: list[str] = POSE_POINTS

class EmbeddingPrediction(BaseModel):
    n_dim: int
    embedding: list[float]

class PredictionResponse(BaseModel):
    pred_type: PredictionType
    image_metadata: ImageMetadata
    prediction: ObjectDetectionPrediction | ClassificationPrediction | PoseDetectionPrediction | EmbeddingPrediction
    