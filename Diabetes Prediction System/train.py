# diabetes_prediction_user_input.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# -------------------------------
# Data Collection and Preprocessing
# -------------------------------
diabetes_dataset = pd.read_csv('diabetes.csv')

# Replace zero values with median for certain columns
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_with_zeros:
    diabetes_dataset[col].replace(0, diabetes_dataset[col].median(), inplace=True)

# Features and Labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
y = diabetes_dataset['Outcome']

# Standardization
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_standardized, y, test_size=0.2, stratify=y, random_state=2
)

# -------------------------------
# Model Training
# -------------------------------
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)

# -------------------------------
# Model Evaluation
# -------------------------------
train_accuracy = accuracy_score(classifier.predict(X_train), y_train)
test_accuracy = accuracy_score(classifier.predict(X_test), y_test)
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

# -------------------------------
# User Input Prediction
# -------------------------------
def get_user_input():
    """
    Get 8 input features from the user.
    Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    """
    print("\nEnter the following patient data:")
    features = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]
    input_values = []
    for feature in features:
        while True:
            try:
                value = float(input(f"{feature}: "))
                input_values.append(value)
                break
            except ValueError:
                print("Invalid input! Please enter a numeric value.")
    return tuple(input_values)

def predict_diabetes(input_data_tuple):
    input_array = np.asarray(input_data_tuple).reshape(1, -1)
    standardized_input = scaler.transform(input_array)
    prediction = classifier.predict(standardized_input)
    return "Diabetic" if prediction[0] == 1 else "Normal"

# Run the user input
user_input = get_user_input()
result = predict_diabetes(user_input)
print(f"\nPrediction for the given input: {result}")
