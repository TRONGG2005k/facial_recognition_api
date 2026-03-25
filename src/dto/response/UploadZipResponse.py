"""
Response model cho batch upload ZIP.
"""

from pydantic import BaseModel, Field


class UploadZipResponse(BaseModel):
    """
    Response khi upload file ZIP đăng ký khuôn mặt hàng loạt.
    
    Attributes:
        error_messages: Danh sách các lỗi nếu có
        total: Tổng số file đã xử lý
        failed_total: Số file bị lỗi
        total_success: Số file xử lý thành công
    """
    error_messages: list[str] = Field(default_factory=list, alias="errorMessage")
    total: int
    failed_total: int = Field(alias="failedTotal")
    total_success: int = Field(alias="totalSuccess")

    class Config:
        populate_by_name = True
