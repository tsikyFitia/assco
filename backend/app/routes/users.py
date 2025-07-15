from fastapi import APIRouter
from app.models.users import Users
from app.database import collection

router = APIRouter()

@router.post("/users")
async def create_user(item: Users):
    result = await collection.insert_one(item.dict())
    return {"id": str(result.inserted_id)}

@router.get("/users")
async def get_items():
    items = []
    async for item in collection.find():
        item["_id"] = str(item["_id"])
        items.append(item)
    return items
