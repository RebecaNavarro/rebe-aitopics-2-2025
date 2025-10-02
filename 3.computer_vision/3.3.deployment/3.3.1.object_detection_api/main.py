from fastapi import FastAPI, UploadFile, File
from tools import get_img_array
from ultralytics import YOLO

app = FastAPI(title="Object detection API")
model = YOLO("yolo11n.pt")

@app.post("/objects")
def detect_objects(file: UploadFile = File(...)):
    img_array = get_img_array(file)
    results = model(img_array)[0]
    labels = [results.names[idx] for idx in results.boxes.cls.tolist()]
    boxes = results.boxes.xyxy.tolist()
    confidences = results.boxes.conf.tolist()

    return {
        "n_detections": len(boxes),
        "boxes": boxes,
        "labels": labels,
        "confidences": confidences
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)