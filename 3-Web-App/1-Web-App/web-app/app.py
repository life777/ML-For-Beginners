import numpy as np
from flask import Flask, request, render_template, redirect
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

#model = pickle.load(open("../ufo-model.pkl", "rb"))
model = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/train")
def trainModelOnTheFly():
    data = pd.read_csv("../data/ufos.csv")
    ufo_data = pd.DataFrame({ "duration": data["duration (seconds)"], "lat": data["latitude"], "lng": data["longitude"], "country": data["country"] })
    ufo_data.dropna(inplace=True)
    ufo_data = ufo_data[(ufo_data.duration <= 60) & (ufo_data.duration >= 1)]

    label_encoder = LabelEncoder()

    x = ufo_data[["duration", "lat", "lng"]]
    y = label_encoder.fit_transform(ufo_data.country)

    global model
    model = LogisticRegression()
    model.fit(x, y)
    return redirect("/")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html")

    int_features = [int(x) for x in request.form.values()]

    # can be usual array
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template(
        "index.html", prediction_text="Likely country: {}".format(countries[output])
    )


if __name__ == "__main__":
    app.run(debug=True)