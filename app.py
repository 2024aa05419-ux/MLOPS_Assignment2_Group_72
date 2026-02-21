import logging
import time
from fastapi import FastAPI, UploadFile, File
import torch
from preprocess import load_image
from model import CNN
import io

app = FastAPI()

logging.basicConfig(level=logging.INFO)

model = CNN()
model.load_state_dict(torch.load("model.pt", map_location=torch.device("cpu")))
model.eval()

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.time()

    contents = await file.read()
    image = load_image(io.BytesIO(contents)).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probability = float(output.item())

    label = "dog" if probability > 0.5 else "cat"

    latency = time.time() - start
    logging.info(f"Prediction latency: {latency:.4f} sec")

    return {
        "probability": probability,
        "label": label
    }