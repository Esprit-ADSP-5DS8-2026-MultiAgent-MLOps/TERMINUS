# Iris Machine Learning Pipeline & REST API

This project demonstrates a complete machine learning lifecycle, from data preprocessing to model deployment as a REST API.

## Project Structure
- `train.py`: Handles data loading, preprocessing (scaling), and training a Random Forest model.
- `main.py`: A FastAPI-based REST API that serves predictions.
- `iris_model.pkl`: The serialized (saved) model pipeline.

## How to Run
1. Install dependencies: 
   `pip install scikit-learn pandas fastapi uvicorn joblib python-multipart`
2. Train the model:
   `python train.py`
3. Start the API:
   `uvicorn main:app --reload`
4. Access the API documentation:
   Open `http://127.0.0.1:8000/docs` in your browser.

## Monitoring Strategy
To prevent performance degradation (Model Rot), the system is designed to monitor for:
- **Data Drift:** Changes in input feature distributions.
- **Concept Drift:** Changes in the relationship between features and labels.
- **Operational Health:** Tracking latency and error rates.
