from fastapi import APIRouter, HTTPException
from app.models.models import Level
from app.database import db  
from bson import ObjectId

router = APIRouter()
collection = db["level"]

@router.post("/level", response_model=Level)
async def create_level(level: Level):
    lev_dict = level.dict(by_alias=True)
    result = await collection.insert_one(lev_dict)
    lev_dict["_id"] = str(result.inserted_id)
    return lev_dict

@router.get("/level", response_model=list[Level])
async def get_level():
    level = []
    async for lev in collection.find():
        lev["_id"] = str(lev["_id"])
        level.append(lev)

    # print(str(subject[0]["_id"]))
    return level

@router.get("/level/{id}", response_model=Level)
async def get_level(id: str):
    level = await collection.find_one({"_id": ObjectId(id)})
    if level:
        level["_id"] = str(level["_id"])
        return level
    raise HTTPException(status_code=404, detail="Level not found")

@router.delete("/level/{id}")
async def delete_level(id: str):
    result = await collection.delete_one({"_id": ObjectId(id)})
    if result.deleted_count == 1:
        return {"message": "Level deleted"}
    raise HTTPException(status_code=404, detail="Level not found")
