import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# 1. Load the public Iris dataset (data about flower petals)
print("Loading data...")
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split data: 80% for training, 20% to test if the model learned correctly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build the Pipeline
# This treats 'Scaling' and 'The Brain' as one single unit
print("Building pipeline...")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100))
])

# 4. Train the model
print("Training the model... please wait.")
pipeline.fit(X_train, y_train)

# 5. Check accuracy
predictions = pipeline.predict(X_test)
score = accuracy_score(y_test, predictions)
print(f"Model Training Complete! Accuracy: {score * 100:.2f}%")

# 6. Save the model to a file
joblib.dump(pipeline, "iris_model.pkl")
print("File saved as: iris_model.pkl")