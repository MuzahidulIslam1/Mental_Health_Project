import os
import pandas as pd
from flask import Flask, render_template, request
from src.models.predict import predict_from_model
from src.utils.logger import get_logger

logger = get_logger("app")



def create_app():
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            try:
                logger.info("Received POST request for prediction.")
                form_data = {key: request.form[key] for key in request.form}
                logger.info(f"Form data: {form_data}")

                df = pd.DataFrame([form_data])
                preds = predict_from_model(df)
                pred_label = "Yes" if preds[0] == 1 else "No"

                logger.info(f"Prediction result: {pred_label}")
                return render_template("result.html", result=pred_label)
            except Exception as e:
                logger.exception(f"Error handling request: {e}")
                return render_template("error.html", error=str(e))
        return render_template("index.html")

    return app
# Create the Flask app instance for Gunicorn
app = create_app()


if __name__ == "__main__":
    app.run(debug=True)

