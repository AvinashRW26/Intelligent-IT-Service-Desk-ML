import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

# Load dataset
data = pd.read_csv("data/tickets.csv")

X = data["ticket_text"]
y = data["category"]

# ML Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultinomialNB())
])

# Train model
model.fit(X, y)

# Save model
with open("ticket_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully")