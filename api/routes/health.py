# Imports

# > External Libraries
from flask import Blueprint
from prometheus_client import generate_latest

health_bp = Blueprint('health', __name__)


@health_bp.route("/health", methods=["GET"])
def health_check() -> tuple[str, int]:
    """
    Health check endpoint for Kubernetes checks

    Returns:
        tuple[str, int]: Response and status code
    """
    return "OK", 200


@health_bp.route("/prometheus", methods=["GET"])
def metrics() -> tuple[str, int]:
    """
    Return the Prometheus metrics for the running flask application

    Returns:
        bytes: Encoded string with the information
    """
    return generate_latest(), 200
