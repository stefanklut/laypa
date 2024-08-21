from flask import Blueprint, request, jsonify, Response
from api.services.prediction_service import process_prediction

predict_bp = Blueprint('predict', __name__)


@predict_bp.route("/predict", methods=["POST"])
def predict() -> tuple[Response, int]:
    """
    Run the prediction on a submitted image

    Returns:
        Response: Submission response

    """
    response_info = process_prediction(request)
    return jsonify(response_info), response_info["status_code"]
