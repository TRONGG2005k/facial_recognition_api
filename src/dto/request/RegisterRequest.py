from pydantic import BaseModel
from typing import List

class RegisterRequest(BaseModel):
    images: List[str]       # danh sách base64
    employee_id: str
