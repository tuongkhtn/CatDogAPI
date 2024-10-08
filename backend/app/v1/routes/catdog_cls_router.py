import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
from fastapi import APIRouter
from fastapi import UploadFile, File

from controllers import Predictor
from schemas.classification import PredictionResponse

from dotenv import load_dotenv
load_dotenv()

DEPLOY_MODEL_NAME = os.getenv("MODEL_NAME")
DEPLOY_MODEL_ALIAS = os.getenv("MODEL_ALIAS")
DEPLOY_DEVICE = os.getenv("DEVICE")

router = APIRouter()
predictor = Predictor(model_name=DEPLOY_MODEL_NAME, model_alias=DEPLOY_MODEL_ALIAS, device=DEPLOY_DEVICE)

@router.post('/predict')
async def predict(file_upload: UploadFile = File(...)):
    response = await predictor.predict(file_upload.file, file_upload.filename)
    return PredictionResponse(**response)
