from ultralytics import YOLO

# Load all trained models
road_model = YOLO("app/ai/models/road_damage.pt")
garbage_model = YOLO("app/ai/models/garbage.pt")
street_model = YOLO("app/ai/models/street_light.pt")
water_model = YOLO("app/ai/models/water_leakage.pt")
tree_model = YOLO("app/ai/models/tree_fallen.pt")
drainage_model = YOLO("app/ai/models/drainage.pt")


def predict(image_path):

    models = [
        ("Road Damage", road_model),
        ("Garbage", garbage_model),
        ("Street Light", street_model),
        ("Water Leakage", water_model),
        ("Tree Fallen", tree_model),
        ("Drainage", drainage_model),
    ]

    best = None
    best_conf = 0.0

    for category, model in models:

        results = model.predict(image_path, verbose=False)

        boxes = results[0].boxes

        if len(boxes) == 0:
            continue

        confidence = float(boxes.conf.max())

        if confidence > best_conf:
            best_conf = confidence
            best = category

    # No issue detected
    if best is None:
        return {
            "category": "Unknown",
            "confidence": 0,
            "department": "General Department",
            "severity": "Low",
        }

    # Department mapping
    department_map = {
        "Road Damage": "Road Department",
        "Garbage": "Sanitation Department",
        "Street Light": "Electricity Department",
        "Water Leakage": "Water Supply Department",
        "Tree Fallen": "Parks Department",
        "Drainage": "Public Works Department",
    }

    # Severity mapping
    if best_conf >= 0.80:
        severity = "High"
    elif best_conf >= 0.50:
        severity = "Medium"
    else:
        severity = "Low"

    return {
        "category": best,
        "confidence": round(best_conf * 100, 2),
        "department": department_map.get(best, "General Department"),
        "severity": severity,
    }