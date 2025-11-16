from fastapi import FastAPI
from app.api.ask import router as ask_router

app = FastAPI(title="Aurora QA API", version="1.0")
app.include_router(ask_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Aurora QA API is running"}
