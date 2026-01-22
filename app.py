from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load trained model
with open("ticket_classifier.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    ticket = data.get("ticket_text")

    prediction = model.predict([ticket])[0]

    return jsonify({
        "ticket_text": ticket,
        "predicted_category": prediction
    })

if __name__ == "__main__":
    app.run(debug=True)