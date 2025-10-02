from fastapi import FastAPI, UploadFile, File
from tools import get_img_array
from inference_models import ObjectDetector, Classificator, PoseDetector

app = FastAPI(title="Object detection API")
object_detector = ObjectDetector()
classificator = Classificator()
pose_detector= PoseDetector()

@app.post("/detect_objects")
def detect_objects(file: UploadFile = File(...)):
    img_array = get_img_array(file)
    return object_detector.predict(img_array)

@app.post("/classify")
def classify(file: UploadFile = File(...)):
    img_array = get_img_array(file)
    return classificator.predict(img_array)

@app.post("/detect_pose")
def detect_pose(file: UploadFile = File(...)):
    img_array = get_img_array(file)
    return pose_detector.predict(img_array)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)