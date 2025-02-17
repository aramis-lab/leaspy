#!/usr/bin/env python3

import json

from flask import Flask, render_template, request

application = Flask(__name__, template_folder="templates")
application._static_folder = "static"


@application.route("/")
def index():
    return render_template("index.html")


@application.route("/model/load", methods=["POST"])
def load_model():
    from models.utils import get_model_derived_parameters

    model = request.get_json()
    model_derived_parameters = get_model_derived_parameters(model)
    return json.dumps(model_derived_parameters, allow_nan=False)


@application.route("/model/personalize", methods=["POST"])
def personalize():
    from models.personalize import get_individual_parameters

    data = request.get_json()
    individual_parameters = get_individual_parameters(data)
    return json.dumps(individual_parameters, allow_nan=False)


if __name__ == "__main__":
    application.run()
