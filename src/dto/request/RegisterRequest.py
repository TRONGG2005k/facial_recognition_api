from pydantic import BaseModel
from typing import List

class RegisterRequest(BaseModel):
    username: str
    images: List[str]       # danh sách base64
    employee_id: str
