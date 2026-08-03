def get_department(category):

    departments = {

        "Road Damage": "Highways Department",

        "Garbage": "Sanitation Department",

        "Street Light": "Electricity Department",

        "Water Leakage": "Water Supply Department",

        "Tree Fallen": "Forest Department",

        "Drainage": "Drainage Department",
    }

    return departments.get(category, "General Department")