from fastapi import FastAPI
from pydantic import BaseModel

from app.pipeline import summarize_document
from fastapi import UploadFile, File

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/summarize")
def summarize(req: TextRequest):
    summary = summarize_document(req.text)
    return {
        "summary": summary
    }

@app.post("/summarize_file")
async def summarize_file(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")

    summary = summarize_document(text)

    return {
        "summary": summary
    }