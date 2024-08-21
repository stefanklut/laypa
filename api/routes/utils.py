# Imports

# > Standard Libraries
from pathlib import Path

# > External Libraries
from flask import Request

# > Project Libraries
from api.services.utils import abort_with_info


def extract_request_fields(request: Request,
                           response_info: dict[str, str]) \
        -> tuple[Request, Path]:
    try:
        identifier = request.form["identifier"]
        response_info["identifier"] = identifier
    except KeyError:
        abort_with_info(400, "Missing identifier in form", response_info)

    try:
        model_name = request.form["model"]
        response_info["model_name"] = model_name
    except KeyError:
        abort_with_info(400, "Missing model in form", response_info)

    try:
        whitelist = request.form.getlist("whitelist")
        response_info["whitelist"] = whitelist
    except KeyError:
        abort_with_info(400, "Missing whitelist in form", response_info)

    try:
        post_file = request.files["image"]
    except KeyError:
        abort_with_info(400, "Missing image in form", response_info)

    if (image_name := post_file.filename) is not None:
        image_name = Path(image_name)
        response_info["filename"] = str(image_name)
    else:
        abort_with_info(400, "Missing filename", response_info)

    return post_file, image_name, identifier, model_name, whitelist
