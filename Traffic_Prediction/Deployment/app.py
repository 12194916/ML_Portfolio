import numpy as np

from flask import Flask, request, jsonify, render_template
import joblib



# Create flask app
flask_app = Flask(__name__)
model = joblib.load(open("model.joblib", "rb"))


@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    features = [[x for x in request.form.values()]]

    features = np.array(features)
    # features = features.reshape(-1, 1)
    prediction = model.predict(features)

    return render_template("index.html", prediction_text = "EXPECTED TRIP DURATION IS {}".format(prediction[0]))


if __name__ == "__main__":
    flask_app.run(debug=True)