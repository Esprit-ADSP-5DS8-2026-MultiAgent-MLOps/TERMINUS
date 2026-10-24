from fastapi import FastAPI
import joblib

# 1. Start the API
app = FastAPI()

# 2. Load the "Brain" you trained in train.py
# If this file name is different in your folder, change it here!
model = joblib.load("iris_model.pkl")

@app.get("/")
def home():
    return {"message": "The Flower Classifier API is live!"}

@app.post("/predict")
def predict(data: list):
    # This takes the 4 numbers you send and asks the model for an answer
    # Example input: [5.1, 3.5, 1.4, 0.2]
    prediction = model.predict([data])
    
    flower_names = ['Setosa', 'Versicolor', 'Virginica']
    result = flower_names[int(prediction[0])]
    
    return {"flower_type": result} 