from insightface.app import FaceAnalysis
import cv2
import numpy as np
import base64
import src.dto.request.RegisterRequest as RegisterRequest
import src.dto.request.RecognizeFaceRequest as RecognizeFaceRequest
from src.db_config.mysqlDb import session
from src.entity.FaceEmbedding import FaceEmbedding
from sqlalchemy.exc import SQLAlchemyError
import uuid

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Tính cosine similarity giữa hai vector."""
    score = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"[DEBUG] Cosine similarity: {score}")
    return score

class FaceRecognitionService:
    def __init__(self):
        print("[DEBUG] Initializing FaceAnalysis model...")
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("[DEBUG] FaceAnalysis model ready")

    # --- Đăng ký face ---
    def register_face(self, request: RegisterRequest) -> str:
        print(f"[DEBUG] Registering face for employee_id={request.employee_id}, images count={len(request.images)}")

        for idx, img_base64 in enumerate(request.images):
            image_data = base64.b64decode(img_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            print(f"[DEBUG] Image {idx} decoded. Is None? {image is None}")

            faces = self.app.get(image)
            print(f"[DEBUG] Faces detected in image {idx}: {len(faces)}")

            if len(faces) == 0:
                raise ValueError(f"No face detected in image index {idx}")

            face_embedding = faces[0].embedding

            new_face = FaceEmbedding(
                id=str(uuid.uuid4()),
                employee_id=request.employee_id,
                embedding=face_embedding.tolist()
            )

            session.add(new_face)   # 🔥 BẮT BUỘC 🔥

        try:
            session.commit()
        except SQLAlchemyError as e:
            print("🔥 DB COMMIT ERROR 🔥")
            print(e)
            session.rollback()
            raise

        print(f"[DEBUG] Commit completed for employee_id={request.employee_id}")
        return "successfully"

    # --- Nhận diện face ---
    def recognize_face(self, request: RecognizeFaceRequest) -> str:
        print("[DEBUG] Recognizing face from image")
        image_data = base64.b64decode(request.image)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        print(f"[DEBUG] Image decoded. Is None? {image is None}")

        faces = self.app.get(image)
        print(f"[DEBUG] Faces detected: {len(faces)}")
        if len(faces) == 0:
            return "No face"

        emb = faces[0].embedding
        all_faces = session.query(FaceEmbedding).all()
        print(f"[DEBUG] Total embeddings in DB: {len(all_faces)}")

        best_match_id = None
        best_score = -1
        for idx, face_record in enumerate(all_faces):
            db_emb = np.array(face_record.embedding)
            score = cosine_similarity(emb, db_emb)
            print(f"[DEBUG] Comparing with DB record {idx} (employee_id={face_record.employee_id}), score={score}")
            if score > best_score:
                best_score = score
                best_match_id = face_record.employee_id

        threshold = 0.6
        if best_score >= threshold:
            print(f"[DEBUG] Best match above threshold: employee_id={best_match_id}, score={best_score}")
            return best_match_id
        else:
            print(f"[DEBUG] No match above threshold ({threshold}), best score={best_score}")
            return None

    # --- Cập nhật embedding cho employee ---
    def update_face(self, employee_id: str, new_images: list[str]) -> str:
        print(f"[DEBUG] Updating face embeddings for employee_id={employee_id}, new_images count={len(new_images)}")
        deleted = session.query(FaceEmbedding).filter_by(employee_id=employee_id).delete()
        session.commit()
        print(f"[DEBUG] Deleted {deleted} old embeddings")

        for idx, img_base64 in enumerate(new_images):
            image_data = base64.b64decode(img_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            print(f"[DEBUG] Update image {idx} decoded. Is None? {image is None}")

            faces = self.app.get(image)
            print(f"[DEBUG] Faces detected in update image {idx}: {len(faces)}")
            if len(faces) == 0:
                continue  # bỏ qua nếu không detect

            face_embedding = faces[0].embedding
            new_face = FaceEmbedding(
                id=str(uuid.uuid4()),
                employee_id=employee_id,
                embedding=face_embedding.tolist()
            )
            session.add(new_face)
            print(f"[DEBUG] Added new embedding for image {idx}")

        session.commit()
        print(f"[DEBUG] Update commit completed for employee_id={employee_id}")
        return "update successfully"

    # --- Xóa tất cả embeddings của employee ---
    def delete_face(self, employee_id: str) -> str:
        deleted = session.query(FaceEmbedding).filter_by(employee_id=employee_id).delete()
        session.commit()
        print(f"[DEBUG] Deleted {deleted} embeddings for employee_id={employee_id}")
        if deleted > 0:
            return "delete successfully"
        else:
            return "no record found"
