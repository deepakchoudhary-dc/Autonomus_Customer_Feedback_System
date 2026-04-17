from fastapi import APIRouter, Request, status

from feedback_system.churn.model import ChurnModel
from feedback_system.churn.schemas import CustomerChurnFeatures, ChurnPredictionResponse

router = APIRouter(prefix="/api/v1/churn", tags=["churn"])


def _get_model(request: Request) -> ChurnModel:
    return request.app.state.churn_model


@router.post("/predict", response_model=ChurnPredictionResponse, status_code=status.HTTP_200_OK)
async def predict_churn(request: Request, payload: CustomerChurnFeatures) -> ChurnPredictionResponse:
    model = _get_model(request)
    probability = model.predict_probability(payload)
    return ChurnPredictionResponse(
        customer_id=payload.customer_id,
        churn_probability=round(probability, 4),
        risk_level=model.risk_level(probability),
    )
