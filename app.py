from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import re

# Load the trained model and vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# FastAPI instance
app = FastAPI()

# Request body structure
class Review(BaseModel):
    review: str

# Preprocess the review text
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only alphabets
    text = text.lower().strip()  # Convert to lowercase and trim
    return text

# Prediction route
@app.post("/predict/")
async def predict_sentiment(review: Review):
    processed_review = preprocess_text(review.review)
    review_tfidf = vectorizer.transform([processed_review])
    prediction = model.predict(review_tfidf)[0]
    print(prediction)
    sentiment = "positive" if prediction == 1 else "negative"
    return {"sentiment": sentiment}

# To run the app, use the command:
# uvicorn app:app --reload


