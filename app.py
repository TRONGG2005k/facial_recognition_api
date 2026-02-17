import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.controller import FaceRecognitionController

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('debug.log')
    ]
)

logger = logging.getLogger(__name__)

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

logger.info("FastAPI app initialized with facial recognition routes")
