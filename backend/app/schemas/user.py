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
    level_id: Optional[str] = None
    guardian_email: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class ImportedUser(BaseModel):
    last_name: str
    first_name: str
    email: EmailStr
    password: Optional[str] = None
    birth_date: Optional[date] = None
    role: str
    institution_id: str
    level_id: Optional[str] = None
    guardian_email: Optional[str] = None

