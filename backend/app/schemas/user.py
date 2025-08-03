from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import date

class UserRegister(BaseModel):
    last_name: str
    first_name: str
    email: EmailStr
    password: str
    birth_date: Optional[date]
    role: str
    institution_id: Optional[str]

class UserLogin(BaseModel):
    email: EmailStr
    password: str
