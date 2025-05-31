from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.server.routers.auth_routes import auth_router
from app.server.routers.creation_routes import creation_router
from app.server.routers.database_routes import database_router
from app.server.routers.image_routes import image_router
from app.server.routers.llm_routes import llm_router
from app.server.routers.me_routes import me_router

app = FastAPI()

# Define the allowed origins
origins = [
    "http://localhost:8080",
    "http://localhost:8081",
    # Add other origins as needed
    "https://buch-ai.github.io",
    "https://bai-buchai-p-run-usea1-server-333749286334.us-east1.run.app",
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows requests from these origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


@app.get("/", tags=["root"])
def root():
    return {"message": "success"}


# Include the routers in the main app with a prefix
app.include_router(auth_router, prefix="/auth")
app.include_router(me_router, prefix="/me")
app.include_router(database_router, prefix="/database")
app.include_router(creation_router, prefix="/creation")
app.include_router(llm_router, prefix="/llm")
app.include_router(image_router, prefix="/image")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
