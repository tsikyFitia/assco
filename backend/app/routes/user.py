from fastapi import APIRouter, HTTPException, Depends, status
from app.models.models import Teacher, Student, Guardian
from app.database import db  
from app.schemas.user import UserRegister, UserLogin
from app.utils.hash import hash_password, verify_password
from app.utils.jwt import create_access_token
from bson import ObjectId
from datetime import datetime, timedelta


from app.auth.auth import get_current_user

router = APIRouter()

collection_user = db["user"]
collection_teacher = db["teacher"]
collection_student = db["student"]
collection_guardian = db["guardian"]


async def create_other_entities(role: str, user_id: str):
    if role.upper() == "TEACHER":
        await collection_teacher.insert_one({"user_id": user_id})
    elif role.upper() == "STUDENT":
        await collection_student.insert_one({"user_id": user_id})
    elif role.upper() == "GUARDIAN":
        await collection_guardian.insert_one({"user_id": user_id})


@router.post("/register")
async def register(user: UserRegister):
    if await collection_user.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email déjà utilisé")

    user_dict = user.dict()

    # Hasher le mot de passe
    user_dict["password_hash"] = hash_password(user_dict.pop("password"))

    # Convertir birth_date si elle existe
    if user_dict.get("birth_date"):
        user_dict["birth_date"] = datetime.combine(user_dict["birth_date"], datetime.min.time())

    result = await collection_user.insert_one(user_dict)
    user_id = str(result.inserted_id)

    await create_other_entities(user.role, user_id)

    return {"message": "Utilisateur créé", "user_id": user_id}


@router.post("/login")
async def login(user: UserLogin):
    found_user = await collection_user.find_one({"email": user.email})
    if not found_user or not verify_password(user.password, found_user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Email ou mot de passe invalide")

    access_token = create_access_token(
        data={"sub": str(found_user["_id"]), "role": found_user["role"]},
        expires_delta=timedelta(minutes=60)
    )

    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    current_user["_id"] = str(current_user["_id"])
    return current_user
