from fastapi import FastAPI
from app.routes.users import router as users_route

app = FastAPI()
app.include_router(users_route)
