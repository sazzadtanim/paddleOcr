import os
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from paddleocr import PaddleOCR
from PIL import Image

app = FastAPI()

# Forced English — not configurable via env var
LANG = "en"
MODEL_DIR = os.getenv("MODEL_DIR", "/models")

ocr = PaddleOCR(
    use_angle_cls=True,
    lang=LANG,
    det_model_dir=f"{MODEL_DIR}/det",
    rec_model_dir=f"{MODEL_DIR}/rec",
    cls_model_dir=f"{MODEL_DIR}/cls",
    show_log=False,
)


@app.get("/health")
def health():
    return {"status": "ok", "lang": LANG}


@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(image)

    result = ocr.ocr(img_array, cls=True)

    output = []
    for line in result[0] if result and result[0] else []:
        box, (text, confidence) = line
        output.append({
            "text": text,
            "confidence": float(confidence),
            "box": box
        })

    return {"results": output}
