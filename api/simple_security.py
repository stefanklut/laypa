import functools
import uuid

import flask
from flask import request, Response, jsonify, Flask
import json


class SimpleSecurity:
    def __init__(self, app: Flask, enabled: bool = False, key_user_json: str = None):
        app.extensions["security"] = self
        self.enabled = enabled
        if enabled:
            self.register_login_resource(app)
            try:
                self.api_key_user = json.loads(key_user_json)
                self.session_key_user = {}
            except Exception as e:
                raise ValueError("When security is enabled, key_user_json should be a valid json string. ", e)

    def is_known_session_key(self, session_key: str):
        return session_key in self.session_key_user.keys()

    def register_login_resource(self, app):
        @app.route("/login", methods=["POST"])
        def login():
            if "Authorization" in request.headers.keys():
                api_key = request.headers["Authorization"]
                session_key = self.login(api_key)

                if session_key is not None:
                    response = Response(status=204)
                    response.headers["X_AUTH_TOKEN"] = session_key

                    return response

            return Response(status=401)

    def login(self, api_key: str) -> str | None:
        if self.enabled and api_key in self.api_key_user:
            session_key = str(uuid.uuid4())
            self.session_key_user[session_key] = self.api_key_user[api_key]
            return session_key

        return None


def session_key_required(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs) -> Response:
        security_ = flask.current_app.extensions["security"]
        if security_.enabled:
            if "Authorization" in request.headers.keys():
                session_key = request.headers["Authorization"]
                if security_.is_known_session_key(session_key):
                    return func(*args, **kwargs)

            response = jsonify({"message": "Expected a valid session key in the Authorization header"})
            response.status_code = 401
            return response
        else:
            print("security disabled")
            return func(*args, **kwargs)

    return decorator
