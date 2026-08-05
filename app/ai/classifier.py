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


def predict(image_path):

    print(f"\nReceived Image: {image_path}")

    models = [
        ("Road Damage", road_model),
        ("Garbage", garbage_model),
        ("Street Light", street_model),
        ("Water Leakage", water_model),
        ("Tree Fallen", tree_model),
        ("Drainage", drainage_model),
    ]

    department_map = {
        "Road Damage": "Road Department",
        "Garbage": "Sanitation Department",
        "Street Light": "Electricity Department",
        "Water Leakage": "Water Supply Department",
        "Tree Fallen": "Parks Department",
        "Drainage": "Public Works Department",
    }

    best_category = None
    best_confidence = 0.0

    try:

        for category, model in models:

            print(f"Running {category} model...")

            start = time.time()

            results = model.predict(
                source=image_path,
                verbose=False,
                conf=0.25,
                imgsz=640,
            )

            end = time.time()

            print(f"{category} completed in {round(end-start,2)} sec")

            if len(results) == 0:
                continue

            if results[0].boxes is None or len(results[0].boxes) == 0:
                print(f"{category}: No Detection")
                continue

            confidence = float(results[0].boxes.conf.max())

            print(f"{category}: {confidence}")

            if confidence > best_confidence:
                best_confidence = confidence
                best_category = category

        if best_category is None:

            print("No Issue Detected")

            return {
                "category": "Unknown",
                "confidence": 0,
                "department": "General Department",
                "severity": "Low",
            }

        if best_confidence >= 0.80:
            severity = "High"
        elif best_confidence >= 0.50:
            severity = "Medium"
        else:
            severity = "Low"

        response = {
            "category": best_category,
            "confidence": round(best_confidence * 100, 2),
            "department": department_map.get(best_category, "General Department"),
            "severity": severity,
        }

        print(response)

        return response

    except Exception as e:
        import traceback

        print("========== AI ERROR ==========")
        traceback.print_exc()

        return {
            "category": "Error",
            "confidence": 0,
            "department": "General Department",
            "severity": "Low",
            "error": str(e),
        }