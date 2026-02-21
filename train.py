import mlflow
import torch

mlflow.set_experiment("catsdogs")

with mlflow.start_run():
    mlflow.log_param("epochs",5)
    accuracy = 0.90
    mlflow.log_metric("accuracy",accuracy)

    torch.save({"model":"demo"},"model.pt")
    mlflow.log_artifact("model.pt")
