from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.predict import router

app = FastAPI(title="CivicAI Backend")

# Allowed Frontend URLs
origins = [
    "http://localhost:5173",
    "https://civic-hub-weld.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routes
app.include_router(router)

# Home Route
@app.get("/")
def home():
    return {
        "message": "CivicAI Backend Running"
    }