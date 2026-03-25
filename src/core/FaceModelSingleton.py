"""
Singleton pattern cho Face Analysis model.
"""

import logging
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "buffalo_l"
CTX_ID = 0  # GPU context
DET_SIZE = (640, 640)  # Detection size


class FaceModelSingleton:
    """
    Thread-safe singleton cho InsightFace model.
    
    Đảm bảo model chỉ được load một lần duy nhất
    trong suốt vòng đờứng dụng.
    
    Example:
        model = FaceModelSingleton.get_model()
        faces = model.get(image)
    """

    _instance: "FaceAnalysis | None" = None
    _lock: Lock = Lock()

    @classmethod
    def get_model(cls) -> "FaceAnalysis":
        """
        Lấy instance của FaceAnalysis model.
        
        Returns:
            FaceAnalysis: Initialized model instance
        """
        if cls._instance is None:
            logger.debug("Initializing FaceModelSingleton")

            with cls._lock:
                # Double-check locking
                if cls._instance is None:
                    cls._instance = cls._create_model()

        return cls._instance

    @classmethod
    def _create_model(cls) -> "FaceAnalysis":
        """
        Tạo và prepare model mới.
        
        Returns:
            FaceAnalysis: Prepared model
        """
        from insightface.app import FaceAnalysis
        
        logger.info(f"Loading InsightFace model: {MODEL_NAME}")
        model = FaceAnalysis(name=MODEL_NAME)

        logger.debug(f"Preparing model with ctx_id={CTX_ID}, det_size={DET_SIZE}")
        model.prepare(ctx_id=CTX_ID, det_size=DET_SIZE)

        logger.info("FaceModelSingleton initialized successfully")
        return model
