from insightface.app import FaceAnalysis
import cv2
import numpy as np
import base64
import src.dto.request.RegisterRequest as RegisterRequest
import src.dto.request.RecognizeFaceRequest as RecognizeFaceRequest
from src.db_config.mysqlDb import session
from src.entity.FaceEmbedding import FaceEmbedding
import uuid

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Tính cosine similarity giữa hai vector."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class FaceRecognitionService:
    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    # --- Đăng ký face ---
    def register_face(self, request: RegisterRequest) -> str:
        for img_base64 in request.images:
            image_data = base64.b64decode(img_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            faces = self.app.get(image)

            if len(faces) == 0:
                raise ValueError("No face detected in one of the images")

            face_embedding = faces[0].embedding
            new_face = FaceEmbedding(
                id=str(uuid.uuid4()),
                employee_id=request.employee_id,
                embedding=face_embedding.tolist()
            )
            session.add(new_face)

        session.commit()
        return "successfully"

    # --- Nhận diện face ---
    def recognize_face(self, request: RecognizeFaceRequest) -> str:
        image_data = base64.b64decode(request.image)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        faces = self.app.get(image)
        if len(faces) == 0:
            return "No face"

        emb = faces[0].embedding
        all_faces = session.query(FaceEmbedding).all()
        best_match_id = None
        best_score = -1
        for face_record in all_faces:
            db_emb = np.array(face_record.embedding)
            score = cosine_similarity(emb, db_emb)
            if score > best_score:
                best_score = score
                best_match_id = face_record.employee_id

        threshold = 0.6
        if best_score >= threshold:
            return best_match_id
        else:
            return None

    # --- Cập nhật embedding cho employee ---
    def update_face(self, employee_id: str, new_images: list[str]) -> str:
        # Xóa embeddings cũ của nhân viên
        session.query(FaceEmbedding).filter_by(employee_id=employee_id).delete()
        session.commit()

        # Thêm embeddings mới
        for img_base64 in new_images:
            image_data = base64.b64decode(img_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            faces = self.app.get(image)

            if len(faces) == 0:
                continue  # bỏ qua nếu không detect

            face_embedding = faces[0].embedding
            new_face = FaceEmbedding(
                id=str(uuid.uuid4()),
                employee_id=employee_id,
                embedding=face_embedding.tolist()
            )
            session.add(new_face)

        session.commit()
        return "update successfully"

    # --- Xóa tất cả embeddings của employee ---
    def delete_face(self, employee_id: str) -> str:
        deleted = session.query(FaceEmbedding).filter_by(employee_id=employee_id).delete()
        session.commit()
        if deleted > 0:
            return "delete successfully"
        else:
            return "no record found"
