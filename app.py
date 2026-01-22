from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load trained model
with open("ticket_classifier.pkl", "rb") as f:
    model = pickle.load(f)

# GUI Route
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        ticket_text = request.form["ticket_text"]
        prediction = model.predict([ticket_text])[0]
    return render_template("index.html", prediction=prediction)

# API Route
@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    ticket = data.get("ticket_text")
    prediction = model.predict([ticket])[0]
    return jsonify({
        "ticket_text": ticket,
        "predicted_category": prediction
    })

if __name__ == "__main__":
    app.run(debug=True)
