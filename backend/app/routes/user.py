from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File
from app.models.models import Teacher, Student, Guardian
from app.database import db
from app.schemas.user import UserRegister, UserLogin
from app.utils.hash import hash_password, verify_password
from app.utils.jwt import create_access_token
from bson import ObjectId
from datetime import datetime, timedelta
from openpyxl import load_workbook
import csv, codecs
from app.auth.auth import get_current_user
from app.utils.helper import create_user_logic

router = APIRouter()

collection_user = db["user"]
collection_teacher = db["teacher"]
collection_student = db["student"]
collection_guardian = db["guardian"]

@router.post("/import-users")
async def import_users(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    if current_user["role"].upper() != "SCHOOL_ADMIN":
        raise HTTPException(403, detail="Seuls les SCHOOL_ADMIN peuvent importer via un fichier.")

    filename = file.filename.lower()

    # Lecture fichier
    if filename.endswith(".csv"):
        reader = csv.DictReader(codecs.iterdecode(file.file, "utf-8"))
        data_rows = list(reader)
    elif filename.endswith((".xlsx", ".xls")):
        wb = load_workbook(filename=file.file)
        ws = wb.active
        rows = list(ws.values)
        header = rows[0]
        data_rows = [dict(zip(header, row)) for row in rows[1:]]
    else:
        raise HTTPException(400, "Format de fichier non supporté (CSV, XLSX, XLS)")

    created, errors = [], []
    roles_order = ["GUARDIAN", "TEACHER", "STUDENT"]

    # Convertir toutes les lignes en objets UserRegister avec ligne d'origine
    all_users = []
    for idx, row in enumerate(data_rows, start=1):
        try:
            user_in = UserRegister(
                email=row["email"],
                last_name=row["last_name"],
                first_name=row["first_name"],
                role=row["role"],
                institution_id=current_user.get("institution_id", ""),
                level_id=row.get("level_id", ""),
                birth_date=row.get("birth_date"),
                password=row.get("password", None),
                guardian_email=row.get("guardian_email", None)
            )
            all_users.append((idx, user_in))
        except Exception as e:
            errors.append({"line": idx, "error": str(e)})

    # Exécuter les insertions par ordre de rôle
    for role in roles_order:
        for line, user in all_users:
            if user.role.upper() == role:
                try:
                    created.append(await create_user_logic(user, current_user))
                except Exception as e:
                    errors.append({"line": line + 1, "error": str(e)})

    return {"message": f"{len(created)} utilisateurs créés", "errors": errors}


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
    user_dict["password_hash"] = hash_password(user_dict.pop("password"))

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
