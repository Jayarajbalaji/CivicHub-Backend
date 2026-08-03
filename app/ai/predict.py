from fastapi import APIRouter, UploadFile, File, Form
import os
import shutil

from app.ai.classifier import predict

router = APIRouter()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@router.post("/predict")
async def predict_issue(
    file: UploadFile = File(...),
    category: str = Form(...)
):

    print("===================================")
    print("Prediction Request Received")
    print("Category:", category)

    ext = os.path.splitext(file.filename)[1]

    if ext == "":
        ext = ".jpg"

    filepath = os.path.join(UPLOAD_FOLDER, f"image{ext}")

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print("Image Saved:", filepath)
    print("Calling AI...")

    result = predict(filepath, category)

    print("AI Finished")
    print(result)

    return result