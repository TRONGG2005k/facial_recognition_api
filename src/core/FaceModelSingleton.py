# core/face_model.py

import logging
from insightface.app import FaceAnalysis
from threading import Lock

logger = logging.getLogger(__name__)


class FaceModelSingleton:
    """
    Thread-safe singleton for InsightFace model
    """

    _instance: FaceAnalysis | None = None
    _lock: Lock = Lock()

    @classmethod
    def get_model(cls) -> FaceAnalysis:

        if cls._instance is None:

            logger.debug("Initializing FaceModelSingleton")

            with cls._lock:

                if cls._instance is None:

                    logger.info("Loading InsightFace model: buffalo_l")
                    model = FaceAnalysis(name="buffalo_l")

                    logger.debug("Preparing model with ctx_id=0, det_size=(640, 640)")
                    model.prepare(
                        ctx_id=0,
                        det_size=(640, 640)
                    )

                    cls._instance = model
                    logger.info("FaceModelSingleton initialized successfully")

        return cls._instance
