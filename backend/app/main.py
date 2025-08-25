from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.routes.user import router as user_route
from app.routes.institution import router as inst_route
from app.routes.subject import router as sub_route
from app.routes.level import router as lev_route
from app.routes.program import router as prog_route
from app.routes.content import router as cont_route
from app.routes.exercise import router as exo_route 
from app.routes.recommend import router as reco_route

app = FastAPI()


# Autoriser le frontend à accéder à l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # ou ["*"] pour tout autoriser (dev seulement)
    allow_credentials=True,
    allow_methods=["*"],  # ou ["GET", "POST", "PUT", "DELETE"] si tu veux être strict
    allow_headers=["*"],
)

# 💡 Inclusion des routes
app.include_router(user_route)
app.include_router(inst_route)
app.include_router(sub_route)
app.include_router(lev_route)
app.include_router(prog_route)
app.include_router(cont_route)
app.include_router(exo_route) 
app.include_router(reco_route) 

# 💡 Modification du schéma OpenAPI pour afficher un champ `Bearer token`
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Mon API",
        version="1.0.0",
        description="API avec authentification par token",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"]["OAuth2PasswordBearer"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
    }

    for path in openapi_schema["paths"].values():
        for method in path.values():
            method["security"] = [{"OAuth2PasswordBearer": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


 #####################################################################################################


#####################################################################################################



# 🧠 On injecte notre schéma customisé
app.openapi = custom_openapi 
