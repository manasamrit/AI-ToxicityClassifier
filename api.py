from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = FastAPI()

class PredictRequest(BaseModel):
    text: str

# Load the Tfidf and model with error handling
try:
    tfidf = pickle.load(open("tf_idf.pkt", "rb"))
    nb_model = pickle.load(open("toxicity_model.pkt", "rb"))
except Exception as e:
    raise RuntimeError(f"Error loading model or vectorizer: {e}")

# Endpoint to check if the API is running
@app.get("/")
async def root():
    return {"message": "Toxicity classifier API is running"}

# Endpoint for prediction
@app.post("/predict")
async def predict(request: PredictRequest):
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="Input text is empty")
    
    # Transform the input to Tfidf vectors
    text_tfidf = tfidf.transform([text])
    
    # Predict the class of the input text
    prediction = nb_model.predict(text_tfidf)
    
    # Map the predicted class to a string
    class_name = "Toxic" if prediction == 1 else "Non-Toxic"
    
    # Return the prediction in a JSON response
    return {
        "text": text,
        "class": class_name
    }
