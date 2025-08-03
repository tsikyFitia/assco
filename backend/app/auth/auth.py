from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from app.utils.jwt import verify_token
from app.database import db


from bson import ObjectId

# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") 
collection_user = db["user"]

async def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user_id = payload.get("sub")
    
    try:
        obj_id = ObjectId(user_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid user ID format")

    user = await collection_user.find_one({"_id": obj_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user

