import cv2
import numpy as np
import base64
import uuid
import logging
from typing import List, Optional

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from src.core.FaceModelSingleton import FaceModelSingleton
from src.entity.FaceEmbedding import FaceEmbedding


logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return -1

    return float(np.dot(a, b) / (norm_a * norm_b))


class FaceRecognitionService:

    def __init__(self):
        logger.debug("Initializing FaceRecognitionService")
        self.app = FaceModelSingleton.get_model()
        logger.debug("FaceRecognitionService initialized with model")

    # ================= INTERNAL =================

    def _extract_embedding(
        self,
        image_bytes: bytes
    ) -> Optional[np.ndarray]:

        logger.debug(f"Extracting embedding from image of size: {len(image_bytes)} bytes")

        nparr = np.frombuffer(image_bytes, np.uint8)

        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            logger.warning("Failed to decode image")
            return None

        faces = self.app.get(image)

        if not faces:
            logger.debug("No faces detected in image")
            return None

        logger.debug(f"Face detected, embedding shape: {faces[0].embedding.shape}")
        return faces[0].embedding

    # ================= REGISTER =================

    def register_embeddings(
        self,
        db: Session,
        employee_id: str,
        images: List[bytes]
    ) -> int:

        logger.debug(f"Registering embeddings for employee_id: {employee_id}, images: {len(images)}")

        embeddings: list[FaceEmbedding] = []

        for img_bytes in images:

            emb = self._extract_embedding(img_bytes)

            if emb is None:
                continue

            embeddings.append(
                FaceEmbedding(
                    id=str(uuid.uuid4()),
                    employee_id=employee_id,
                    embedding=emb.tolist()
                )
            )

        if not embeddings:
            logger.warning(f"No valid embeddings extracted for employee_id: {employee_id}")
            return 0

        try:

            db.bulk_save_objects(embeddings)
            logger.debug(f"Saved {len(embeddings)} embeddings for employee_id: {employee_id}")

            return len(embeddings)

        except SQLAlchemyError:

            logger.error(
                f"DB error while saving embeddings for employee_id: {employee_id}",
                exc_info=True
            )

            raise

    def register_from_base64(
        self,
        db: Session,
        employee_id: str,
        base64_images: List[str]
    ) -> int:

        images = [
            base64.b64decode(img)
            for img in base64_images
        ]

        return self.register_embeddings(
            db,
            employee_id,
            images
        )

    # ================= RECOGNIZE =================

    def recognize(
        self,
        db: Session,
        base64_image: str,
        threshold: float = 0.6
    ) -> Optional[str]:

        logger.debug(f"Starting face recognition with threshold: {threshold}")

        image_bytes = base64.b64decode(base64_image)

        emb = self._extract_embedding(image_bytes)

        if emb is None:
            logger.warning("Failed to extract embedding for recognition")
            return None

        best_score = -1
        best_employee = None

        query = db.query(
            FaceEmbedding.employee_id,
            FaceEmbedding.embedding
        ).yield_per(1000)

        count = 0
        for row in query:
            count += 1
            db_emb = np.array(row.embedding)

            score = cosine_similarity(emb, db_emb)

            if score > best_score:
                best_score = score
                best_employee = row.employee_id

        logger.debug(f"Compared against {count} embeddings, best score: {best_score}")

        if best_score >= threshold:
            logger.info(f"Recognition successful: employee_id {best_employee}, score {best_score}")
            return best_employee

        logger.info(f"Recognition failed: best score {best_score} below threshold {threshold}")
        return None

    # ================= UPDATE =================

    def update(
        self,
        db: Session,
        employee_id: str,
        base64_images: List[str]
    ) -> int:

        db.query(FaceEmbedding).filter_by(
            employee_id=employee_id
        ).delete()

        images = [
            base64.b64decode(img)
            for img in base64_images
        ]

        return self.register_embeddings(
            db,
            employee_id,
            images
        )

    # ================= DELETE =================

    def delete(
        self,
        db: Session,
        employee_id: str
    ) -> int:

        return db.query(FaceEmbedding).filter_by(
            employee_id=employee_id
        ).delete()
