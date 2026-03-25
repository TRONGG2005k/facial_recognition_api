"""
Entity model cho face embeddings.
"""

import uuid

from sqlalchemy import Column, String, TIMESTAMP, text
from sqlalchemy.dialects.mysql import JSON

from src.db_config import Base


class FaceEmbedding(Base):
    """
    Model lưu trữ face embeddings.
    
    Attributes:
        id: UUID của bản ghi
        employee_id: ID nhân viên
        embedding: Vector embedding dạng JSON
        created_at: Thờing tạo
        updated_at: Thờing cập nhật
    """
    
    __tablename__ = "face_embeddings"

    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )
    employee_id = Column(
        String(36),
        nullable=False,
        index=True  # Thêm index để query nhanh hơn
    )
    embedding = Column(JSON, nullable=False)

    created_at = Column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP")
    )
    updated_at = Column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP")
    )

    def __repr__(self) -> str:
        return f"<FaceEmbedding(id={self.id}, employee_id={self.employee_id})>"
