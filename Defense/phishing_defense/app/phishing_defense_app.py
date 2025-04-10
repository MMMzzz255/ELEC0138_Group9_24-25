from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
# Change the model here: random forest or SVM
model = joblib.load("../model/rf_model.pkl")
imputer = joblib.load("../model/rf_imputer.pkl")
top_features = joblib.load("../model/rf_top_features.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main route for the web application. It handles both GET and POST requests.
    :return:
        Rendered HTML template with input form and prediction result.
    """
    prediction = None
    if request.method == "POST":
        try:
            input_data = [float(request.form[feature]) for feature in top_features]
            input_df = pd.DataFrame([input_data], columns=top_features)
            input_imputed = pd.DataFrame(imputer.transform(input_df), columns=top_features)
            pred = model.predict(input_imputed)[0]
            prediction = "Phishing Website Detected!" if pred == 1 else "Safe Website"
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", features=top_features, prediction=prediction)

if __name__ == "__main__":
    """
    Run the Flask application.
    """
    app.run(port=5000)
