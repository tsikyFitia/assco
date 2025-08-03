from fastapi import APIRouter, HTTPException
from app.models.models import Institution
from app.database import db 
from bson import ObjectId

router = APIRouter()
collection = db["institution"]

@router.post("/institution", response_model=Institution)
async def create_institution(institution: Institution):
    inst_dict = institution.dict(by_alias=True)
    result = await collection.insert_one(inst_dict)
    inst_dict["_id"] = str(result.inserted_id)
    return inst_dict

@router.get("/institution", response_model=list[Institution])
async def get_institutions():
    institutions = []
    async for inst in collection.find():
        # print(str(inst))
        inst["_id"] = str(inst["_id"])
        # print(inst["_id"])
        institutions.append(inst)

    # print(str(institutions[0]["_id"]))
    return institutions

@router.get("/institution/{id}", response_model=Institution)
async def get_institution(id: str):
    institution = await collection.find_one({"_id": ObjectId(id)})
    if institution:
        institution["_id"] = str(institution["_id"])
        return institution
    raise HTTPException(status_code=404, detail="Institution not found")

@router.delete("/institution/{id}")
async def delete_institution(id: str):
    result = await collection.delete_one({"_id": ObjectId(id)})
    if result.deleted_count == 1:
        return {"message": "Institution deleted"}
    raise HTTPException(status_code=404, detail="Institution not found")
