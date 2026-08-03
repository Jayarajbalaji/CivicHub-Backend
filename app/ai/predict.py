from fastapi import APIRouter, UploadFile, File
import os
import shutil

from app.ai.classifier import predict

router = APIRouter()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@router.post("/predict")
async def predict_issue(file: UploadFile = File(...)):

    print("Received filename:", file.filename)

    ext = os.path.splitext(file.filename)[1]

    if ext == "":
        ext = ".jpg"

    filepath = os.path.join(UPLOAD_FOLDER, f"image{ext}")

    print("Saving file to:", filepath)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print("File exists:", os.path.exists(filepath))

    result = predict(filepath)

    return result