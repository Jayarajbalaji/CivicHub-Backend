from ultralytics import YOLO
import time

print("Loading AI Models...")

road_model = YOLO("app/ai/models/road_damage.pt")
garbage_model = YOLO("app/ai/models/garbage.pt")
street_model = YOLO("app/ai/models/street_light.pt")
water_model = YOLO("app/ai/models/water_leakage.pt")
tree_model = YOLO("app/ai/models/tree_fallen.pt")
drainage_model = YOLO("app/ai/models/drainage.pt")

print("All AI Models Loaded Successfully")


MODEL_MAP = {
    "Road Damage": road_model,
    "Garbage": garbage_model,
    "Street Light": street_model,
    "Water Leakage": water_model,
    "Tree Fallen": tree_model,
    "Drainage": drainage_model,
}


DEPARTMENT_MAP = {
    "Road Damage": "Road Department",
    "Garbage": "Sanitation Department",
    "Street Light": "Electricity Department",
    "Water Leakage": "Water Supply Department",
    "Tree Fallen": "Parks Department",
    "Drainage": "Public Works Department",
}


def predict(image_path, category):

    print("================================")
    print("Selected Category :", category)

    model = MODEL_MAP.get(category)

    if model is None:
        return {
            "category": "Unknown",
            "confidence": 0,
            "department": "General Department",
            "severity": "Low",
        }

    start = time.time()

    results = model.predict(
        source=image_path,
        verbose=False,
        conf=0.25,
        imgsz=640,
    )

    end = time.time()

    print(f"Inference Time : {round(end-start,2)} sec")

    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:

        print("No Detection")

        return {
            "category": category,
            "confidence": 0,
            "department": DEPARTMENT_MAP.get(category),
            "severity": "Low",
        }

    confidence = float(results[0].boxes.conf.max())

    if confidence >= 0.80:
        severity = "High"
    elif confidence >= 0.50:
        severity = "Medium"
    else:
        severity = "Low"

    response = {
        "category": category,
        "confidence": round(confidence * 100, 2),
        "department": DEPARTMENT_MAP.get(category),
        "severity": severity,
    }

    print(response)

    return response