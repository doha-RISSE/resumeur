from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.pipeline import summarize_document
app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/summarize")
def summarize(req: TextRequest):
    return {"summary": summarize_document(req.text)}

@app.post("/summarize_file")
async def summarize_file(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")

    return {"summary": summarize_document(text)}