from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse

app = FastAPI(title="CERAai ERP Setup Copilot - MVP")

security = HTTPBasic()

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/secure-check")
def secure_check(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username == "demo" and credentials.password == "demo":
        return {"access": "granted"}
    raise HTTPException(status_code=401, detail="Unauthorized")
