from fastapi import FastAPI
from routes import router as StudentRouter

app = FastAPI()
app.include_router(StudentRouter, tags=["Student"], prefix="/student")

@app.get("/", tags=["Root"])
async def root():
    return {"message": "Success"}