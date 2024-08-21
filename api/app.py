# Imports

# > Standard Libraries
import logging

# > External Libraries
from flask import Flask

# > Project Libraries
from api.routes.predict import predict_bp
from api.routes.health import health_bp
from api.setup.environment import read_environment_variables
from api.setup.initialization import initialize_environment
from api.services.model_setup import PredictorGenPageWrapper

from main import setup_logging
from utils.logging_utils import get_logger_name


def create_app():
    # Capture logging
    setup_logging()
    logger = logging.getLogger(get_logger_name())
    logger.info("Starting Laypa API")

    # Read environment variables
    logger.info("Initializing environment")
    max_queue_size, model_base_path, output_base_path = read_environment_variables()
    args, executor, queue_size_gauge, images_processed_counter, exception_predict_counter = initialize_environment(
        max_queue_size, output_base_path
    )
    logger.info("Environment initialized successfully")

    # Initialize model
    logger.info("Initializing model wrapper")
    predict_gen_page_wrapper = PredictorGenPageWrapper(model_base_path)
    logger.info("Model wrapper initialized successfully")

    app = Flask(__name__)

    app.register_blueprint(predict_bp)
    app.register_blueprint(health_bp)

    app.executor = executor
    app.args = args
    app.predict_gen_page_wrapper = predict_gen_page_wrapper
    app.output_base_path = output_base_path
    app.images_processed_counter = images_processed_counter
    app.max_queue_size = max_queue_size

    logger.info("Laypa API started successfully")

    return app


if __name__ == "__main__":
    app = create_app()
