from pydantic import BaseModel
from typing import Optional

class Users(BaseModel):
    name: str
    lastname: Optional[str] = None
