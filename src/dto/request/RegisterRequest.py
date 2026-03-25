"""
Request model cho đăng ký khuôn mặt.
"""

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    """
    Request đăng ký khuôn mặt mới.
    
    Attributes:
        images: Danh sách ảnh base64 của khuôn mặt
        employee_id: ID nhân viên cần đăng ký
    """
    images: list[str] = Field(
        ...,
        min_length=1,
        description="Danh sách ảnh base64, ít nhất 1 ảnh"
    )
    employee_id: str = Field(
        ...,
        min_length=1,
        description="ID nhân viên"
    )
