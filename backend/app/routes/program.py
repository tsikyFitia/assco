from fastapi import APIRouter, HTTPException
from app.models.models import Program
from app.database import db  
from bson import ObjectId

router = APIRouter()
collection = db["program"]

@router.post("/program", response_model=Program)
async def create_program(program: Program):
    prog_dict = program.dict(by_alias=True)
    result = await collection.insert_one(prog_dict)
    prog_dict["_id"] = str(result.inserted_id)
    return prog_dict

@router.get("/program", response_model=list[Program])
async def get_program():
    program = []
    async for prog in collection.find():
        prog["_id"] = str(prog["_id"])
        program.append(prog)

    # print(str(subject[0]["_id"]))
    return program

@router.get("/program/{id}", response_model=Program)
async def get_program(id: str):
    program = await collection.find_one({"_id": ObjectId(id)})
    if program:
        program["_id"] = str(program["_id"])
        return program
    raise HTTPException(status_code=404, detail="Program not found")

@router.delete("/program/{id}")
async def delete_program(id: str):
    result = await collection.delete_one({"_id": ObjectId(id)})
    if result.deleted_count == 1:
        return {"message": "Program deleted"}
    raise HTTPException(status_code=404, detail="Program not found")
