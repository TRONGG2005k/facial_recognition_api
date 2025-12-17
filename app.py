from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.controller import FaceRecognitionController
app = FastAPI()

# --- CORS middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc list domain frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include router từ controller
app.include_router(FaceRecognitionController.router, prefix="/facial-recognition")
