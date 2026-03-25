"""
Controller xử lý các API endpoints cho nhận diện khuôn mặt.
"""

import logging
import time
from typing import Callable, TypeVar
from functools import wraps

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.service import FaceRecognitionService, FaceBatchService
from src.dto.request import RegisterRequest, RecognizeFaceRequest
from src.db_config import get_db

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Face Recognition"])

# Initialize services
_face_service = FaceRecognitionService()
_batch_service = FaceBatchService(_face_service)


# ============================
# Request Models
# ============================

class UpdateFaceRequest(BaseModel):
    """Request cập nhật khuôn mặt."""
    employee_id: str = Field(..., min_length=1)
    new_images: list[str] = Field(..., min_length=1)


class DeleteFaceRequest(BaseModel):
    """Request xóa khuôn mặt."""
    employee_id: str = Field(..., min_length=1)


# ============================
# Response Models
# ============================

class SuccessResponse(BaseModel):
    """Response thành công chung."""
    message: str


class RegisterResponse(SuccessResponse):
    """Response đăng ký khuôn mặt."""
    embeddings: int


class DeleteResponse(SuccessResponse):
    """Response xóa khuôn mặt."""
    deleted: int


class RecognizeResponse(BaseModel):
    """Response nhận diện khuôn mặt."""
    message: str | None = None
    employee_id: str | None = None


# ============================
# Decorators
# ============================

T = TypeVar("T")


def handle_db_transaction(
    func: Callable[..., T]
) -> Callable[..., T]:
    """
    Decorator xử lý database transaction.
    
    Tự động commit nếu thành công, rollback nếu có lỗi.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        db: Session | None = kwargs.get("db")
        try:
            result = func(*args, **kwargs)
            if db:
                db.commit()
            return result
        except HTTPException:
            if db:
                db.rollback()
            raise
        except Exception as e:
            if db:
                db.rollback()
            logger.error(f"Database error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    return wrapper


def log_endpoint(action: str):
    """
    Decorator log thông tin endpoint.
    
    Args:
        action: Mô tả action đang thực hiện
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"{action} endpoint called")
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================
# Endpoints
# ============================

@router.get("/hello")
@log_endpoint("Hello")
def hello() -> dict[str, str]:
    """Endpoint test kết nối."""
    return {"message": "Hello world"}


@router.post("/register-face-batch")
@log_endpoint("Batch register")
async def register_face_batch(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> dict:
    """
    Đăng ký khuôn mặt hàng loạt từ file ZIP.
    
    Cấu trúc ZIP: employee_id/FACE/image.jpg
    """
    if not file.filename or not file.filename.lower().endswith(".zip"):
        logger.warning(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="File must be ZIP")

    try:
        result = _batch_service.register_from_zip(db, file.file)
        db.commit()
        logger.info(f"Batch register completed: {result}")
        return result.model_dump(by_alias=True)
    except Exception as e:
        logger.error(f"Batch register failed: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/register-face", response_model=RegisterResponse)
@log_endpoint("Register face")
@handle_db_transaction
def register_face(
    request: RegisterRequest,
    db: Session = Depends(get_db)
) -> RegisterResponse:
    """Đăng ký khuôn mặt mới cho nhân viên."""
    logger.debug(
        f"Registering face for employee: {request.employee_id}, "
        f"images: {len(request.images)}"
    )
    
    count = _face_service.register_from_base64(
        db,
        request.employee_id,
        request.images
    )
    
    logger.info(f"Registered {count} embeddings for {request.employee_id}")
    return RegisterResponse(
        message="Registered successfully",
        embeddings=count
    )


@router.post("/face-recognition", response_model=RecognizeResponse)
@log_endpoint("Face recognition")
def face_recognition(
    request: RecognizeFaceRequest,
    db: Session = Depends(get_db)
) -> RecognizeResponse:
    """Nhận diện khuôn mặt."""
    employee_id = _face_service.recognize(db, request.image)
    
    if employee_id is None:
        return RecognizeResponse(message=None)
    
    return RecognizeResponse(employee_id=employee_id)


@router.put("/update-face", response_model=RegisterResponse)
@log_endpoint("Update face")
@handle_db_transaction
def update_face(
    request: UpdateFaceRequest,
    db: Session = Depends(get_db)
) -> RegisterResponse:
    """Cập nhật khuôn mặt cho nhân viên."""
    logger.debug(
        f"Updating face for employee: {request.employee_id}, "
        f"images: {len(request.new_images)}"
    )
    
    count = _face_service.update(
        db,
        request.employee_id,
        request.new_images
    )
    
    logger.info(f"Updated {count} embeddings for {request.employee_id}")
    return RegisterResponse(
        message="Updated successfully",
        embeddings=count
    )


@router.delete("/delete-face", response_model=DeleteResponse)
@log_endpoint("Delete face")
@handle_db_transaction
def delete_face(
    request: DeleteFaceRequest,
    db: Session = Depends(get_db)
) -> DeleteResponse:
    """Xóa khuôn mặt của nhân viên."""
    logger.debug(f"Deleting face for employee: {request.employee_id}")
    
    deleted = _face_service.delete(db, request.employee_id)
    
    logger.info(f"Deleted {deleted} embeddings for {request.employee_id}")
    return DeleteResponse(
        message="Deleted successfully",
        deleted=deleted
    )
