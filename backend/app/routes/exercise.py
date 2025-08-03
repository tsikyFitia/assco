from fastapi import APIRouter, HTTPException
from app.models.models import Exercise
from app.database import db 
from bson import ObjectId

router = APIRouter()
collection = db["exercise"]

@router.post("/exercise", response_model=Exercise)
async def create_exercise(exercise: Exercise):
    exo_dict = exercise.dict(by_alias=True)
    result = await collection.insert_one(exo_dict)
    exo_dict["_id"] = str(result.inserted_id)
    return exo_dict

@router.get("/exercise", response_model=list[Exercise])
async def get_exercise():
    exercise = []
    async for exo in collection.find():
        # print(str(inst))
        exo["_id"] = str(exo["_id"])
        # print(inst["_id"])
        exercise.append(exo)

    # print(str(institutions[0]["_id"]))
    return exercise

@router.get("/exercise/{id}", response_model=Exercise)
async def get_exercise(id: str):
    exercise = await collection.find_one({"_id": ObjectId(id)})
    if exercise:
        exercise["_id"] = str(exercise["_id"])
        return exercise
    raise HTTPException(status_code=404, detail="Exercise not found")

@router.delete("/exercise/{id}")
async def delete_exerciset(id: str):
    result = await collection.delete_one({"_id": ObjectId(id)})
    if result.deleted_count == 1:
        return {"message": "Exercise deleted"}
    raise HTTPException(status_code=404, detail="Exercise not found")
