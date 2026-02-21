import logging
import time
import io
from fastapi import FastAPI, UploadFile, File, HTTPException

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
# Global Model Variable
# ----------------------------
model = None
torch = None
load_image = None


# ----------------------------
# Load Model at Startup
# ----------------------------
@app.on_event("startup")
def load_model():
    global model, torch, load_image

    logger.info("Starting application...")

    try:
        import torch as torch_lib
        from preprocess import load_image as load_img
        from model import CNN

        torch = torch_lib
        load_image = load_img

        model_instance = CNN()
        model_instance.load_state_dict(
            torch.load("model.pt", map_location=torch.device("cpu"))
        )
        model_instance.eval()

        model = model_instance

        logger.info("Model loaded successfully.")

    except Exception as e:
        logger.warning(f"Model not loaded: {e}")
        model = None


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
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")

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
        raise HTTPException(status_code=500, detail="Prediction failed")