from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from src.service.FaceRecognitionService import FaceRecognitionService
from src.dto.request.RegisterRequest import RegisterRequest
from src.dto.request.RecognizeFaceRequest import RecognizeFaceRequest
router = APIRouter()
face_service = FaceRecognitionService()

class UpdateFaceRequest(BaseModel):
    employee_id: str
    new_images: List[str]

class DeleteFaceRequest(BaseModel):
    employee_id: str

# --- Endpoints ---
@router.get("/hello")
def hello():
    return {"message": "Hello world"}

@router.post("/register-face")
def register_face(request: RegisterRequest):
    try:
        result = face_service.register_face(request)
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/face-recognition")
def face_recognition(request: RecognizeFaceRequest):
    result = face_service.recognize_face(request)
    if result is None:
        return {"message": "No match found"}
    return {"employee_id": result}

@router.put("/update-face")
def update_face(request: UpdateFaceRequest):
    try:
        result = face_service.update_face(request.employee_id, request.new_images)
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/delete-face")
def delete_face(request: DeleteFaceRequest):
    try:
        result = face_service.delete_face(request.employee_id)
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
