"""
Request model cho nhận diện khuôn mặt.
"""

from pydantic import BaseModel, Field


class RecognizeFaceRequest(BaseModel):
    """
    Request nhận diện khuôn mặt.
    
    Attributes:
        image: Ảnh base64 cần nhận diện
    """
    image: str = Field(
        ...,
        min_length=1,
        description="Ảnh base64 cần nhận diện"
    )
