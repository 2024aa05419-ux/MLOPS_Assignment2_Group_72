import mlflow
import torch
from model import CNN

mlflow.set_experiment("catsdogs")

model = CNN()

with mlflow.start_run():
    mlflow.log_param("epochs", 5)

    accuracy = 0.90
    mlflow.log_metric("accuracy", accuracy)

    torch.save(model.state_dict(), "model.pt")
    mlflow.log_artifact("model.pt")