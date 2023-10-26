import os

from flask_app import app
from gunicorn.app.base import BaseApplication


class GunicornApp(BaseApplication):
    """
    Gunicorn wrapper around the existing flask app
    """

    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        for key, value in self.options.items():
            self.cfg.set(key, value)

    def load(self):
        return self.application


if __name__ == "__main__":
    # Run gunicorn based on environment variables
    try:
        bind = os.environ["GUNICORN_RUN_HOST"]
        workers = os.environ["GUNICORN_WORKERS"]
        threads = os.environ["GUNICORN_THREADS"]
        accesslog = os.environ["GUNICORN_ACCESSLOG"]
    except KeyError as error:
        raise KeyError(f"Missing Gunicorn Environment variable: {error.args[0]}")

    options = {
        "bind": bind,
        "workers": workers,
        "threads": threads,
        "accesslog": accesslog,
    }

    gunicorn_app = GunicornApp(app, options)
    gunicorn_app.run()
