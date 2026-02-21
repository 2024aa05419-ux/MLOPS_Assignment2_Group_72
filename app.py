import logging
import time
import io
from fastapi import FastAPI, UploadFile, File
import torch
from preprocess import load_image
from model import CNN

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("catsdogs-api")

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="Cats vs Dogs Classifier API")

# ----------------------------
# Load Model
# ----------------------------
logger.info("Loading model...")

model = CNN()
model.load_state_dict(torch.load("model.pt", map_location=torch.device("cpu")))
model.eval()

logger.info("Model loaded successfully.")

# ----------------------------
# Health Endpoint
# ----------------------------
@app.get("/health")
def health():
    logger.info("Health check endpoint called.")
    return {"status": "healthy"}

# ----------------------------
# Prediction Endpoint
# ----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()

    logger.info(f"Received file: {file.filename}")

    try:
        contents = await file.read()
        image = load_image(io.BytesIO(contents)).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            probability = float(output.item())

        label = "dog" if probability > 0.5 else "cat"

        latency = time.time() - start_time

        logger.info(
            f"Prediction successful | "
            f"Label: {label} | "
            f"Probability: {probability:.4f} | "
            f"Latency: {latency:.4f} sec"
        )

        return {
            "probability": probability,
            "label": label,
            "latency_sec": round(latency, 4)
        }

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {"error": "Prediction failed"}