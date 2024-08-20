from pydantic import BaseModel

class PredictionResponse(BaseModel):
    probs: list = []
    best_prob: float = -1.0
    predicted_id: int = -1
    predicted_class: str = ""
    predicted_name: str = ""
    predicted_alias: str = ""