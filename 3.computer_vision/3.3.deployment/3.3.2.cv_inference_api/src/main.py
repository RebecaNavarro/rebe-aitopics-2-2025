from fastapi import FastAPI, UploadFile, File
from src.tools import get_img_array
from src.inference_models import ObjectDetector, Classificator, PoseDetector, ImageEmbedder
from src.schemas import PredictionResponse, PredictionType, ImageMetadata

app = FastAPI(title="Object detection API")
object_detector = ObjectDetector(model_name="/app/models/yolo11n.pt")
classificator = Classificator(model_name="/app/models/yolo11n-cls.pt")
pose_detector= PoseDetector(model_name="/app/models/yolo11n-pose.pt")
image_embedder = ImageEmbedder(model_path="/app/models/embeddings_model.tflite")

@app.post("/detect_objects")
def detect_objects(file: UploadFile = File(...)) -> PredictionResponse:
    img_array = get_img_array(file)
    prediction = object_detector.predict(img_array)
    return PredictionResponse(
        pred_type=PredictionType.object_detection,
        image_metadata=ImageMetadata(file_name=file.filename, height=img_array.shape[0], width=img_array.shape[1], channels=img_array.shape[2]),
        prediction=prediction
    )

@app.post("/classify")
def classify(file: UploadFile = File(...)) -> PredictionResponse:
    img_array = get_img_array(file)
    prediction = classificator.predict(img_array)
    return PredictionResponse(
        pred_type=PredictionType.classification,
        image_metadata=ImageMetadata(file_name=file.filename, height=img_array.shape[0], width=img_array.shape[1], channels=img_array.shape[2]),
        prediction=prediction
    )

@app.post("/detect_pose")
def detect_pose(file: UploadFile = File(...)) -> PredictionResponse:
    img_array = get_img_array(file)
    prediction = pose_detector.predict(img_array)
    return PredictionResponse(
        pred_type=PredictionType.pose_detection,
        image_metadata=ImageMetadata(file_name=file.filename, height=img_array.shape[0], width=img_array.shape[1], channels=img_array.shape[2]),
        prediction=prediction
    )

@app.post("/compute_embedding")
def compute_embeddings(file: UploadFile = File(...)) -> PredictionResponse:
    img_array = get_img_array(file)
    prediction = image_embedder.predict(img_array)
    return PredictionResponse(
        pred_type=PredictionType.embedding,
        image_metadata=ImageMetadata(file_name=file.filename, height=img_array.shape[0], width=img_array.shape[1], channels=img_array.shape[2]),
        prediction=prediction
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)