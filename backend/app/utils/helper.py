from app.database import db
from app.utils.hash import hash_password
from datetime import datetime

collection_user = db["user"]
collection_teacher = db["teacher"]
collection_student = db["student"]
collection_guardian = db["guardian"]
collection_level = db["level"]
collection_teaching = db["teaching"]

async def send_email(to_email: str, subject: str, content: str):
    print(f"[EMAIL] À: {to_email} | Sujet: {subject} \n{content}")

async def find_level_id_by_name(level_name: str, institution_id: str):
    print(f"🔍 Recherche du niveau: '{level_name}' pour institution {institution_id}")
    level_doc = await collection_level.find_one({
        "name": {"$regex": f"^{level_name}$", "$options": "i"},
        "institution_id": institution_id
    })
    if not level_doc:
        raise Exception(f"Niveau '{level_name}' introuvable dans institution {institution_id}")
    level_id = str(level_doc["_id"])
    print(f"✅ Niveau trouvé: {level_id}")
    return level_id

async def create_user_logic(user, creator_user):
    print("🟢 Début de création de l'utilisateur")
    print(f"📧 Email: {user.email}")
    
    if await collection_user.find_one({"email": user.email}):
        raise Exception(f"Email déjà utilisé: {user.email}")
    print("✅ Email non utilisé")

    # Récupère institution_id depuis user ou creator_user
    institution_id = str(user.institution_id or creator_user.get("institution_id"))
    print(f"🏫 Institution ID utilisé: {institution_id}")
    if not institution_id:
        raise Exception("institution_id manquant pour l'utilisateur")

    # Construction du dictionnaire utilisateur
    user_dict = user.dict(exclude={"password", "guardian_email", "level_id"})
    user_dict["institution_id"] = institution_id
    user_dict["password_hash"] = hash_password(user.password or "TempPass123!")
    print("🔐 Mot de passe hashé")

    # Date de naissance
    if user.birth_date:
        print(f"📅 Date de naissance reçue: {user.birth_date}")
        if isinstance(user.birth_date, str):
            user_dict["birth_date"] = datetime.strptime(user.birth_date, "%Y-%m-%d")
        else:
            user_dict["birth_date"] = datetime.combine(user.birth_date, datetime.min.time())
        print(f"📅 Date de naissance formatée: {user_dict['birth_date']}")

    # Insertion dans la collection `user`
    result = await collection_user.insert_one(user_dict)
    user_id = str(result.inserted_id)
    role = user.role.upper()
    print(f"👤 Utilisateur inséré avec ID: {user_id} | Rôle: {role}")

    # Traitement par rôle
    if role == "GUARDIAN":
        print("👪 Création du guardian")
        await collection_guardian.insert_one({"user_id": user_id})
        print("✅ Guardian inséré")

    elif role == "TEACHER":
        print("🎓 Création du teacher")
        await collection_teacher.insert_one({"user_id": user_id})
        level_names = [name.strip() for name in (user.level_id or "").split(";") if name.strip()]
        print(f"📚 Niveaux à associer au teacher: {level_names}")
        for level_name in level_names:
            level_id = await find_level_id_by_name(level_name, institution_id)
            await collection_teaching.insert_one({
                "teacher_id": user_id,
                "institution_id": institution_id,
                "level_id": level_id
            })
            print(f"✅ Association teacher-niveau: {level_id}")

    elif role == "STUDENT":
        print("🧒 Création du student")
        print(f"📋 Student info: {user}")
        level_name = (user.level_id or "").strip()
        print(f"🎓 Nom du niveau étudiant: {level_name}")
        if not level_name:
            raise Exception("Niveau obligatoire pour un étudiant")
        level_id = await find_level_id_by_name(level_name, institution_id)

        print("🧹 Suppression des doublons student")
        await collection_student.delete_many({"user_id": user_id})

        guardian_emails = [e.strip() for e in (user.guardian_email or "").split(";") if e.strip()]
        print(f"📨 Emails des gardiens trouvés: {guardian_emails}")
        guardian_ids = []

        for email in guardian_emails:
            guardian = await collection_user.find_one({"email": email, "role": "GUARDIAN"})
            if not guardian:
                raise Exception(f"Guardian non trouvé : {email}")
            guardian_id = str(guardian["_id"])
            guardian_ids.append(guardian_id)
            print(f"✅ Guardian trouvé: {guardian_id}")

        for guardian_id in guardian_ids:
            await collection_student.insert_one({
                "user_id": user_id,
                "level_id": level_id,
                "guardian_id": guardian_id
            })
            print(f"✅ Étudiant lié à guardian {guardian_id} avec niveau {level_id}")

    else:
        print("⚠️ Rôle non reconnu (aucune action spécifique)")

    await send_email(
        to_email=user.email,
        subject="Compte activé",
        content=f"Bonjour {user.first_name}, votre compte a été créé. Mot de passe: {'TempPass123!' if not user.password else '****'}"
    )

    print("✅ Email de confirmation envoyé")
    print("✅ Fin de création de l'utilisateur\n")

    return {"user_id": user_id, "email": user.email, "role": user.role}
