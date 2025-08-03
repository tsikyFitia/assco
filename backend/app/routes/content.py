from fastapi import APIRouter, HTTPException
from app.models.models import Content
from app.database import db 
from bson import ObjectId

router = APIRouter()
collection = db["content"]

@router.post("/content", response_model=Content)
async def create_content(content: Content):
    cont_dict = content.dict(by_alias=True)
    result = await collection.insert_one(cont_dict)
    cont_dict["_id"] = str(result.inserted_id)
    return cont_dict

@router.get("/content", response_model=list[Content])
async def get_content():
    content = []
    async for inst in collection.find():
        # print(str(inst))
        cont["_id"] = str(cont["_id"])
        # print(inst["_id"])
        content.append(cont)

    # print(str(institutions[0]["_id"]))
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
