from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import cv2
import numpy as np
from tools import get_img_array
from tarea_inference_models import ObjectDetector, Classificator, PoseDetector

app = FastAPI(title="Object detection API")
object_detector = ObjectDetector()
classificator = Classificator()
pose_detector = PoseDetector()

def numpy_to_bytes(img_array: np.ndarray) -> bytes:
    _, encoded_img = cv2.imencode('.jpg', img_array)
    return encoded_img.tobytes()

@app.post("/detect_objects")
def detect_objects(file: UploadFile = File(...)):
    img_array = get_img_array(file)
    annotated_img = object_detector.predict_with_image(img_array)
    img_bytes = numpy_to_bytes(annotated_img)
    return Response(content=img_bytes, media_type="image/jpeg")

@app.post("/classify")
def classify(file: UploadFile = File(...)):
    img_array = get_img_array(file)
    annotated_img = classificator.predict_with_image(img_array)
    img_bytes = numpy_to_bytes(annotated_img)
    return Response(content=img_bytes, media_type="image/jpeg")

@app.post("/detect_pose")
def detect_pose(file: UploadFile = File(...)):
    img_array = get_img_array(file)
    annotated_img = pose_detector.predict_with_image(img_array)
    img_bytes = numpy_to_bytes(annotated_img)
    return Response(content=img_bytes, media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)