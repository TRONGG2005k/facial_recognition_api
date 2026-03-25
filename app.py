"""
FastAPI application chính cho hệ thống nhận diện khuôn mặt.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.controller import FaceRecognitionController
from src.db_config import init_db

# ============================
# Logging Configuration
# ============================


def configure_logging() -> None:
    """Cấu hình logging cho ứng dụng."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("debug.log"),
        ],
    )


configure_logging()
logger = logging.getLogger(__name__)


# ============================
# Lifespan Events
# ============================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events cho FastAPI app.
    
    Chạy init_db() khi app start.
    """
    logger.info("Starting up...")
    init_db()
    yield
    logger.info("Shutting down...")


# ============================
# App Configuration
# ============================


def create_app() -> FastAPI:
    """
    Factory function tạo FastAPI app.

    Returns:
        FastAPI: Configured application instance
    """
    app = FastAPI(
        title="Face Recognition API",
        description="API nhận diện và quản lý khuôn mặt",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(
        FaceRecognitionController.router,
        prefix="/facial-recognition",
        tags=["Face Recognition"],
    )

    logger.info("FastAPI app initialized with facial recognition routes")
    return app


# Create app instance
app = create_app()


# Health check endpoint
@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}
