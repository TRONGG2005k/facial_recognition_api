from src.db_config.mysqlDb import Base
from sqlalchemy import Column, String, TIMESTAMP, text
from sqlalchemy.dialects.mysql import JSON
import uuid

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    employee_id = Column(String(36), nullable=False)
    embedding = Column(JSON, nullable=False)

    created_at = Column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP")
    )
    updated_at = Column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP")
    )

    def __repr__(self):
        return f"<FaceEmbedding id={self.id} employee_id={self.employee_id}>"