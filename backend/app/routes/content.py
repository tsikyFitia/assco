""" from fastapi import APIRouter, HTTPException, Form, File, Depends, UploadFile
from app.models.models import Content
from app.database import db 
from bson import ObjectId
from app.auth.auth import get_current_user
from datetime import datetime
import os
import shutil
import uuid

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from app.models.models import Exercise

from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

router = APIRouter()
collection = db["content"]

exercise_collection = db["exercise"]

UPLOAD_FOLDER = "uploads/pdf"

@router.post("/content", response_model=Content)
async def create_content(content: Content):
    cont_dict = content.dict(by_alias=True)
    result = await collection.insert_one(cont_dict)
    cont_dict["_id"] = str(result.inserted_id)
    return cont_dict


@router.get("/content", response_model=list[Content])
async def get_all_content():
    content = []
    async for inst in collection.find():
        inst["_id"] = str(inst["_id"])
        content.append(inst)
    return content


@router.get("/content/{id}", response_model=Content)
async def get_content(id: str):
    content = await collection.find_one({"_id": ObjectId(id)})
    if content:
        content["_id"] = str(content["_id"])
        return content
    raise HTTPException(status_code=404, detail="Content not found")


@router.delete("/content/{id}")
async def delete_content(id: str):
    result = await collection.delete_one({"_id": ObjectId(id)})
    if result.deleted_count == 1:
        return {"message": "Content deleted"}
    raise HTTPException(status_code=404, detail="Content not found")


@router.post("/content/upload")
async def upload_content(
    file: UploadFile = File(...),
    title: str = Form(...),
    subject_id: str = Form(...),
    level_id: str = Form(...),
    type: str = Form(...),
    format: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Seuls les fichiers PDF sont acceptés")

    # 🆔 Générer un nom de fichier unique
    file_id = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, file_id)

    # 📁 Créer le dossier si nécessaire
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 💾 Sauvegarder le fichier
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 📝 Créer le document Content
    content_doc = {
        "title": title,
        "type": type,
        "format": format,
        "file_id": file_id,
        "uploaded_at": datetime.utcnow(),
        "validated_by_admin": False,
        "subject_id": subject_id,
        "level_id": level_id,
        "teacher_id": str(current_user["_id"]),
        "institution_id": str(current_user["institution_id"]),
    }

    result = await collection.insert_one(content_doc)

    return {
        "message": "Contenu uploadé avec succès",
        "content_id": str(result.inserted_id),
        "file_id": file_id
    }






# Chargement des modèles au démarrage
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

tokenizer = AutoTokenizer.from_pretrained("iarfmoose/t5-base-question-generator")
model = AutoModelForSeq2SeqLM.from_pretrained("iarfmoose/t5-base-question-generator")


# Fonctions génération questions
def generate_questions_t5(text, num_questions=5):
    input_text = "generate questions: " + text
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = t5_model.generate(inputs, max_length=150, num_return_sequences=num_questions, num_beams=max(5, num_questions), no_repeat_ngram_size=2)
    return [t5_tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

def generate_questions_gpt2(text, num_questions=5):
    # 🧠 Définir pad_token si nécessaire
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    # 🧾 Tokenisation avec padding + attention_mask
    inputs = gpt2_tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    )

    # 🤖 Génération du texte
    outputs = gpt2_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=100,
        num_return_sequences=num_questions,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        pad_token_id=gpt2_tokenizer.pad_token_id
    )

    # 🧾 Décodage des questions générées
    return [gpt2_tokenizer.decode(out, skip_special_tokens=True) for out in outputs]



def generate_questions_bart(text, num_questions=5):
    input_text = "Generate questions: " + text
    inputs = bart_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = bart_model.generate(inputs, max_length=150, num_return_sequences=num_questions, num_beams=max(5, num_questions), no_repeat_ngram_size=2)
    return [bart_tokenizer.decode(out, skip_special_tokens=True) for out in outputs]





def generate_questions_tuned(text, num_questions=3):
    inputs = tokenizer.encode(
        "generate questions: " + text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    outputs = model.generate(
        inputs,
        max_length=128,
        num_beams=10,
        num_return_sequences=num_questions,
        early_stopping=True
    )

    return [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]


# Route modifiée
@router.post("/exercise/from-content/{content_id}", response_model=Exercise)
async def generate_exercise_from_content(
    content_id: str,
    model_name: str = Form("autre"),
    num_questions: int = Form(5),
    current_user: dict = Depends(get_current_user)
):
    content = await collection.find_one({"_id": ObjectId(content_id)})
    if not content:
        raise HTTPException(404, detail="Contenu non trouvé")

    file_path = os.path.join(UPLOAD_FOLDER, content["file_id"])
    if not os.path.exists(file_path):
        raise HTTPException(404, detail="Fichier PDF introuvable")

    pdf_text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                pdf_text += page.get_text()
                images = page.get_images(full=True)
                for img_info in images:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    text_from_image = pytesseract.image_to_string(image)
                    pdf_text += "\n" + text_from_image
    except Exception as e:
        raise HTTPException(500, detail=f"Erreur lors de l'extraction: {str(e)}")

    if not pdf_text.strip():
        raise HTTPException(400, detail="Le contenu PDF est vide")

    # Choix du modèle et génération
    model_name = model_name.lower()
    if model_name == "t5":
        questions = generate_questions_t5(pdf_text, num_questions)
    elif model_name == "gpt2":
        questions = generate_questions_gpt2(pdf_text, num_questions)
    elif model_name == "bart":
        questions = generate_questions_bart(pdf_text, num_questions)
    else:
        questions = generate_questions_tuned(pdf_text, num_questions)

    exercise_doc = {
        "title": f"Exercice auto-généré - {content['title']} [{model_name.upper()}]",
        "evaluation_type": "auto-generated",
        "author": "AI",
        "questions": questions,
        "created_at": datetime.utcnow(),
        "subject_id": content["subject_id"],
        "level_id": content["level_id"],
        "creator_id": str(current_user["_id"]),
        "institution_id": str(current_user["institution_id"]),
    }

    result = await exercise_collection.insert_one(exercise_doc)
    exercise_doc["_id"] = str(result.inserted_id)

    return exercise_doc """




""" from fastapi import APIRouter, HTTPException, Form, Depends
from app.database import db
from app.auth.auth import get_current_user
from app.models.models import Exercise, Question
from bson import ObjectId
from datetime import datetime

import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import json

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

router = APIRouter()
collection = db["content"]
exercise_collection = db["exercise"]

UPLOAD_FOLDER = "uploads/pdf"

# 🔹 Charger le modèle Flan-T5 open-source
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def chunk_text(text, max_tokens=400):
    words = text.split()
    chunks = []
    current = []
    for w in words:
        current.append(w)
        if len(current) >= max_tokens:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


def generate_questions_from_chunk(chunk, num_questions=3):
    prompt = (
        f"Generate {num_questions} multiple choice questions from the following text:\n{chunk}\n"
        "Format: JSON array with objects containing 'question', 'options', 'answer'"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model.generate(
        **inputs,
        max_length=256,
        num_return_sequences=1,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )
    text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        q_parsed = json.loads(text_output)
        if not isinstance(q_parsed, list):
            raise ValueError("JSON n'est pas une liste")
        questions = []
        for q in q_parsed:
            questions.append({
                "question_text": str(q.get("question") or q.get("question_text") or "Question non disponible"),
                "options": q.get("options") if isinstance(q.get("options"), list) else [],
                "answer": str(q.get("answer") or "")
            })
        return questions
    except Exception as e:
        # Log pour debug si le modèle a renvoyé du texte non JSON
        print(f"[Warning] Chunk JSON parse failed: {e}\nRaw output: {text_output[:200]}...")
        # fallback : texte brut
        return [{
            "question_text": text_output[:512],  # limiter la taille
            "options": [],
            "answer": ""
        }]



@router.post("/exercise/from-content/{content_id}", response_model=Exercise)
async def generate_exercise_from_content(
    content_id: str,
    num_questions: int = Form(5),
    current_user: dict = Depends(get_current_user)
):
    content = await collection.find_one({"_id": ObjectId(content_id)})
    if not content:
        raise HTTPException(404, detail="Contenu non trouvé")

    file_path = os.path.join(UPLOAD_FOLDER, content["file_id"])
    if not os.path.exists(file_path):
        raise HTTPException(404, detail="Fichier PDF introuvable")

    # Extraction texte et images
    pdf_text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                pdf_text += page.get_text()
                for img_info in page.get_images(full=True):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    pdf_text += "\n" + pytesseract.image_to_string(image)
    except Exception as e:
        raise HTTPException(500, detail=f"Erreur lors de l'extraction: {str(e)}")

    if not pdf_text.strip():
        raise HTTPException(400, detail="Le contenu PDF est vide")

    # Chunking du texte
    chunks = chunk_text(pdf_text, max_tokens=400)

    # Génération questions
    questions_structured = []
    for chunk in chunks:
        chunk_questions = generate_questions_from_chunk(chunk, num_questions)
        for q in chunk_questions:
            questions_structured.append(
                Question(
                    question_text=q.get("question", q.get("question_text", "")),
                    options=q.get("options", []),
                    answer=str(q.get("answer") or "")
                )
            )

    questions_structured = questions_structured[:num_questions]

    # Créer l'exercice en dict pour insertion Mongo
    exercise_data = {
        "title": f"Exercice auto-généré - {content['title']}",
        "evaluation_type": "auto-generated",
        "author": "AI",
        "questions": [q.dict() for q in questions_structured],
        "created_at": datetime.utcnow(),
        "subject_id": content["subject_id"],
        "level_id": content["level_id"],
        "creator_id": str(current_user["_id"]),
        "institution_id": str(current_user["institution_id"]),
    }

    # 🔹 Insertion MongoDB
    result = await exercise_collection.insert_one(exercise_data)

    # 🔹 Récupérer l'exercice depuis MongoDB (avec _id natif)
    exercise_from_db = await exercise_collection.find_one({"_id": result.inserted_id})

    # 🔹 Transformer en Exercise Pydantic (ignorer _id)
    exercise_instance = Exercise(
        title=exercise_from_db["title"],
        evaluation_type=exercise_from_db["evaluation_type"],
        author=exercise_from_db["author"],
        questions=[Question(**q) for q in exercise_from_db["questions"]],
        created_at=exercise_from_db["created_at"],
        subject_id=exercise_from_db["subject_id"],
        level_id=exercise_from_db["level_id"],
        creator_id=exercise_from_db["creator_id"],
        institution_id=exercise_from_db["institution_id"]
    )

    return exercise_instance """



from fastapi import APIRouter, HTTPException, Form, Depends
from app.database import db
from app.auth.auth import get_current_user
from app.models.models import Exercise, Question
from bson import ObjectId
from datetime import datetime

import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import json
import requests

router = APIRouter()
collection = db["content"]
exercise_collection = db["exercise"]

UPLOAD_FOLDER = "uploads/pdf"
HUGGINGFACE_TOKEN = "ton_token_hf"  # 🔹 Remplace par ton token Hugging Face
MISTRAL_MODEL_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

# ==========================
# Fonctions utilitaires
# ==========================
def chunk_text(text, max_tokens=400):
    words = text.split()
    chunks = []
    current = []
    for w in words:
        current.append(w)
        if len(current) >= max_tokens:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks

def generate_questions_from_chunk(chunk, num_questions=3):
    prompt = f"""
Tu es un assistant pédagogique. À partir du texte suivant, génère {num_questions} questions QCM.
Chaque question doit avoir :
- 1 bonne réponse
- 3 mauvaises réponses
- Indique la bonne réponse

Texte :
{chunk[:4000]}  # limiter la longueur du prompt

Format JSON :
[
  {{
    "question": "Question ?",
    "options": ["A", "B", "C", "D"],
    "correct_answer": "A"
  }}
]
"""

    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 800,
            "temperature": 0.7
        }
    }

    response = requests.post(MISTRAL_MODEL_URL, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"[Error HuggingFace API] {response.status_code} - {response.text}")
        return [{
            "question_text": "Erreur lors de la génération",
            "options": [],
            "answer": ""
        }]

    try:
        result_text = response.json()[0]["generated_text"]
        q_parsed = json.loads(result_text)
        questions = []
        for q in q_parsed:
            questions.append({
                "question_text": str(q.get("question") or q.get("question_text") or "Question non disponible"),
                "options": q.get("options") if isinstance(q.get("options"), list) else [],
                "answer": str(q.get("correct_answer") or "")
            })
        return questions
    except Exception as e:
        print(f"[Warning] JSON parse failed: {e}\nRaw output: {response.text[:200]}...")
        return [{
            "question_text": response.text[:512],
            "options": [],
            "answer": ""
        }]

# ==========================
# Route FastAPI
# ==========================
@router.post("/exercise/from-content/{content_id}", response_model=Exercise)
async def generate_exercise_from_content(
    content_id: str,
    num_questions: int = Form(5),
    current_user: dict = Depends(get_current_user)
):
    content = await collection.find_one({"_id": ObjectId(content_id)})
    if not content:
        raise HTTPException(404, detail="Contenu non trouvé")

    file_path = os.path.join(UPLOAD_FOLDER, content["file_id"])
    if not os.path.exists(file_path):
        raise HTTPException(404, detail="Fichier PDF introuvable")

    # Extraction texte et images
    pdf_text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                pdf_text += page.get_text()
                for img_info in page.get_images(full=True):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    pdf_text += "\n" + pytesseract.image_to_string(image)
    except Exception as e:
        raise HTTPException(500, detail=f"Erreur lors de l'extraction: {str(e)}")

    if not pdf_text.strip():
        raise HTTPException(400, detail="Le contenu PDF est vide")

    # Chunking du texte
    chunks = chunk_text(pdf_text, max_tokens=400)

    # Génération questions
    questions_structured = []
    for chunk in chunks:
        chunk_questions = generate_questions_from_chunk(chunk, num_questions)
        for q in chunk_questions:
            questions_structured.append(
                Question(
                    question_text=q.get("question_text", ""),
                    options=q.get("options", []),
                    answer=q.get("answer", "")
                )
            )

    questions_structured = questions_structured[:num_questions]

    # Préparer l'exercice pour MongoDB
    exercise_data = {
        "title": f"Exercice auto-généré - {content['title']}",
        "evaluation_type": "auto-generated",
        "author": "AI",
        "questions": [q.dict() for q in questions_structured],
        "created_at": datetime.utcnow(),
        "subject_id": content["subject_id"],
        "level_id": content["level_id"],
        "creator_id": str(current_user["_id"]),
        "institution_id": str(current_user["institution_id"]),
    }

    result = await exercise_collection.insert_one(exercise_data)
    exercise_from_db = await exercise_collection.find_one({"_id": result.inserted_id})

    exercise_instance = Exercise(
        title=exercise_from_db["title"],
        evaluation_type=exercise_from_db["evaluation_type"],
        author=exercise_from_db["author"],
        questions=[Question(**q) for q in exercise_from_db["questions"]],
        created_at=exercise_from_db["created_at"],
        subject_id=exercise_from_db["subject_id"],
        level_id=exercise_from_db["level_id"],
        creator_id=exercise_from_db["creator_id"],
        institution_id=exercise_from_db["institution_id"]
    )

    return exercise_instance


