from pydantic import BaseModel

class RecognizeFaceRequest(BaseModel):
    image: str
