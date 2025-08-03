from fastapi import APIRouter, HTTPException
from app.models.models import Subject
from app.database import db  
from bson import ObjectId

router = APIRouter()
collection = db["subject"]

@router.post("/subject", response_model=Subject)
async def create_subject(subject: Subject):
    sub_dict = subject.dict(by_alias=True)
    result = await collection.insert_one(sub_dict)
    sub_dict["_id"] = str(result.inserted_id)
    return sub_dict

@router.get("/subject", response_model=list[Subject])
async def get_subject():
    subject = []
    async for sub in collection.find():
        # print(str(sub))
        sub["_id"] = str(sub["_id"])
        # print(sub["_id"])
        subject.append(sub)

    # print(str(subject[0]["_id"]))
    return subject

@router.get("/subject/{id}", response_model=Subject)
async def get_subject(id: str):
    subject = await collection.find_one({"_id": ObjectId(id)})
    if subject:
        subject["_id"] = str(subject["_id"])
        return subject
    raise HTTPException(status_code=404, detail="Subject not found")

@router.delete("/subject/{id}")
async def delete_subject(id: str):
    result = await collection.delete_one({"_id": ObjectId(id)})
    if result.deleted_count == 1:
        return {"message": "Subject deleted"}
    raise HTTPException(status_code=404, detail="Subject not found")
