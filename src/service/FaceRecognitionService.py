"""
Service xử lý nhận diện và đăng ký khuôn mặt.
"""

import base64
import logging
import time
import uuid
from typing import TYPE_CHECKING

import cv2
import numpy as np
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.core import FaceModelSingleton
from src.entity import FaceEmbedding

if TYPE_CHECKING:
    from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

# Constants
DEFAULT_THRESHOLD = 0.6
BATCH_SIZE = 1000


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Tính độ tương đồng cosine giữa 2 vector.
    
    Args:
        a: Vector thứ nhất
        b: Vector thứ hai
        
    Returns:
        Giá trị similarity từ -1 đến 1
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return -1.0

    return float(np.dot(a, b) / (norm_a * norm_b))


class FaceRecognitionService:
    """
    Service nhận diện và quản lý khuôn mặt.
    
    Cung cấp các chức năng:
    - Đăng ký khuôn mặt mới
    - Nhận diện khuôn mặt
    - Cập nhật/Xóa khuôn mặt
    """

    def __init__(self) -> None:
        logger.debug("Initializing FaceRecognitionService")
        self._model: "FaceAnalysis" = FaceModelSingleton.get_model()
        logger.debug("FaceRecognitionService initialized")

    # ============================
    # Internal Methods
    # ============================

    def _extract_embedding(self, image_bytes: bytes) -> np.ndarray | None:
        """
        Trích xuất embedding từ ảnh.
        
        Args:
            image_bytes: Dữ liệu ảnh dạng bytes
            
        Returns:
            Embedding vector hoặc None nếu không tìm thấy khuôn mặt
        """
        logger.debug(f"Extracting embedding from {len(image_bytes)} bytes")

        # Decode ảnh
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            logger.warning("Failed to decode image")
            return None

        # Detect khuôn mặt
        faces = self._model.get(image)

        if not faces:
            logger.debug("No faces detected in image")
            return None

        logger.debug(f"Face detected, embedding shape: {faces[0].embedding.shape}")
        return faces[0].embedding

    def _decode_base64_images(self, base64_images: list[str]) -> list[bytes]:
        """
        Decode danh sách ảnh base64.
        
        Args:
            base64_images: Danh sách ảnh base64
            
        Returns:
            Danh sách ảnh dạng bytes
        """
        return [base64.b64decode(img) for img in base64_images]

    # ============================
    # Registration
    # ============================

    def register_embeddings(
        self,
        db: Session,
        employee_id: str,
        images: list[bytes]
    ) -> int:
        """
        Đăng ký embeddings cho nhân viên.
        
        Args:
            db: Database session
            employee_id: ID nhân viên
            images: Danh sách ảnh dạng bytes
            
        Returns:
            Số embeddings đã lưu thành công
            
        Raises:
            SQLAlchemyError: Nếu có lỗi database
        """
        logger.debug(
            f"Registering embeddings for {employee_id}, "
            f"images: {len(images)}"
        )

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
            logger.warning(f"No valid embeddings for {employee_id}")
            return 0

        try:
            db.bulk_save_objects(embeddings)
            logger.debug(f"Saved {len(embeddings)} embeddings for {employee_id}")
            return len(embeddings)
        except SQLAlchemyError:
            logger.error(f"DB error saving embeddings for {employee_id}", exc_info=True)
            raise

    def register_from_base64(
        self,
        db: Session,
        employee_id: str,
        base64_images: list[str]
    ) -> int:
        """
        Đăng ký từ ảnh base64.
        
        Args:
            db: Database session
            employee_id: ID nhân viên
            base64_images: Danh sách ảnh base64
            
        Returns:
            Số embeddings đã lưu thành công
        """
        images = self._decode_base64_images(base64_images)
        return self.register_embeddings(db, employee_id, images)

    # ============================
    # Recognition
    # ============================

    def recognize(
        self,
        db: Session,
        base64_image: str,
        threshold: float = DEFAULT_THRESHOLD
    ) -> str | None:
        """
        Nhận diện khuôn mặt.
        
        Args:
            db: Database session
            base64_image: Ảnh cần nhận diện (base64)
            threshold: Ngưỡng similarity (mặc định 0.6)
            
        Returns:
            employee_id nếu tìm thấy, None nếu không
        """
        start_time = time.perf_counter()

        image_bytes = base64.b64decode(base64_image)
        emb = self._extract_embedding(image_bytes)

        if emb is None:
            return None

        best_score = -1.0
        best_employee: str | None = None
        comparison_count = 0

        # Đo thởi gian truy vấn DB
        db_start = time.perf_counter()
        query = (
            db.query(FaceEmbedding.employee_id, FaceEmbedding.embedding)
            .yield_per(BATCH_SIZE)
        )

        for row in query:
            comparison_count += 1
            db_emb = np.array(row.embedding)
            score = cosine_similarity(emb, db_emb)

            if score > best_score:
                best_score = score
                best_employee = row.employee_id
        
        db_time = time.perf_counter() - db_start
        total_time = time.perf_counter() - start_time

        logger.info(f"Recognition: db_query_time={db_time:.3f}s, total_time={total_time:.3f}s")

        if best_score >= threshold:
            return best_employee

        return None

    # ============================
    # Update & Delete
    # ============================

    def update(
        self,
        db: Session,
        employee_id: str,
        base64_images: list[str]
    ) -> int:
        """
        Cập nhật khuôn mặt (xóa cũ + thêm mới).
        
        Args:
            db: Database session
            employee_id: ID nhân viên
            base64_images: Danh sách ảnh mới (base64)
            
        Returns:
            Số embeddings đã thêm mới
        """
        # Xóa embeddings cũ
        deleted = db.query(FaceEmbedding).filter_by(employee_id=employee_id).delete()
        logger.debug(f"Deleted {deleted} old embeddings for {employee_id}")

        # Thêm embeddings mới
        images = self._decode_base64_images(base64_images)
        return self.register_embeddings(db, employee_id, images)

    def delete(self, db: Session, employee_id: str) -> int:
        """
        Xóa tất cả khuôn mặt của nhân viên.
        
        Args:
            db: Database session
            employee_id: ID nhân viên
            
        Returns:
            Số embeddings đã xóa
        """
        return db.query(FaceEmbedding).filter_by(employee_id=employee_id).delete()
