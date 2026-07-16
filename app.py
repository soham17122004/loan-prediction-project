from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
with open("models/loan_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Change these according to your model
        features = [
            float(request.form["Gender"]),
            float(request.form["Married"]),
            float(request.form["Dependents"]),
            float(request.form["Education"]),
            float(request.form["Self_Employed"]),
            float(request.form["ApplicantIncome"]),
            float(request.form["CoapplicantIncome"]),
            float(request.form["LoanAmount"]),
            float(request.form["Loan_Amount_Term"]),
            float(request.form["Credit_History"]),
            float(request.form["Property_Area"]),
        ]

        features = np.array(features).reshape(1, -1)

        features = scaler.transform(features)

        prediction = model.predict(features)

        if prediction[0] == 1:
            result = "✅ Loan Approved"
        else:
            result = "❌ Loan Rejected"

    except Exception as e:
        result = str(e)

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)
