import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List

from src.service.FaceRecognitionService import FaceRecognitionService
from src.service.FaceModelSingleton import FaceBatchService
from src.dto.request.RegisterRequest import RegisterRequest
from src.dto.request.RecognizeFaceRequest import RecognizeFaceRequest
from src.db_config.mysqlDb import get_db

logger = logging.getLogger(__name__)

router = APIRouter()

face_service = FaceRecognitionService()
batch_service = FaceBatchService(face_service)


class UpdateFaceRequest(BaseModel):
    employee_id: str
    new_images: List[str]


class DeleteFaceRequest(BaseModel):
    employee_id: str


# ================= HELLO =================

@router.get("/hello")
def hello():
    logger.debug("Hello endpoint called")
    return {"message": "Hello world"}


# ================= BATCH REGISTER =================

@router.post("/register-face-batch")
async def register_face_batch(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    logger.debug(f"Batch register called with file: {file.filename}")

    try:

        if not file.filename.lower().endswith(".zip"):
            logger.warning(f"Invalid file type: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail="File must be ZIP"
            )

        result = batch_service.register_from_zip(
            db,
            file.file
        )

        db.commit()

        logger.info(f"Batch register completed: {result}")
        return result

    except Exception as e:

        logger.error(f"Batch register failed: {str(e)}", exc_info=True)
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ================= REGISTER =================

@router.post("/register-face")
def register_face(
    request: RegisterRequest,
    db: Session = Depends(get_db)
):

    logger.debug(f"Register face called for employee_id: {request.employee_id}, images: {len(request.images)}")

    try:

        count = face_service.register_from_base64(
            db,
            request.employee_id,
            request.images
        )

        db.commit()

        logger.info(f"Register face completed for {request.employee_id}: {count} embeddings")
        return {
            "message": "Registered successfully",
            "embeddings": count
        }

    except Exception as e:

        logger.error(f"Register face failed for {request.employee_id}: {str(e)}", exc_info=True)
        db.rollback()

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# ================= RECOGNIZE =================

@router.post("/face-recognition")
def face_recognition(
    request: RecognizeFaceRequest,
    db: Session = Depends(get_db)
):

    logger.debug("Face recognition called")

    try:

        employee_id = face_service.recognize(
            db,
            request.image
        )

        if employee_id is None:
            logger.info("Face recognition: No match found")
            return {"message": "No match found"}

        logger.info(f"Face recognition: Matched employee_id: {employee_id}")
        return {"employee_id": employee_id}

    except Exception as e:

        logger.error(f"Face recognition failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ================= UPDATE =================

@router.put("/update-face")
def update_face(
    request: UpdateFaceRequest,
    db: Session = Depends(get_db)
):

    logger.debug(f"Update face called for employee_id: {request.employee_id}, images: {len(request.new_images)}")

    try:

        count = face_service.update(
            db,
            request.employee_id,
            request.new_images
        )

        db.commit()

        logger.info(f"Update face completed for {request.employee_id}: {count} embeddings")
        return {
            "message": "Updated successfully",
            "embeddings": count
        }

    except Exception as e:

        logger.error(f"Update face failed for {request.employee_id}: {str(e)}", exc_info=True)
        db.rollback()

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# ================= DELETE =================

@router.delete("/delete-face")
def delete_face(
    request: DeleteFaceRequest,
    db: Session = Depends(get_db)
):

    logger.debug(f"Delete face called for employee_id: {request.employee_id}")

    try:

        deleted = face_service.delete(
            db,
            request.employee_id
        )

        db.commit()

        logger.info(f"Delete face completed for {request.employee_id}: {deleted} embeddings deleted")
        return {
            "message": "Deleted successfully",
            "deleted": deleted
        }

    except Exception as e:

        logger.error(f"Delete face failed for {request.employee_id}: {str(e)}", exc_info=True)
        db.rollback()

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
