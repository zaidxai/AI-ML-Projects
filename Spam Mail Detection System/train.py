# Advanced SMS Spam Detection System
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset
data = pd.read_csv("mail_data.csv")

# Replace null values
data = data.fillna("")

# 2. Encode Labels
# spam = 0, ham = 1
data['Category'] = data['Category'].map({'spam': 0, 'ham': 1})

# 3. Features and Labels
X = data['Message']
y = data['Category']

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=3
)

# 5. TF-IDF Feature Extraction (Improved)
vectorizer = TfidfVectorizer(
    min_df=2,
    stop_words='english',
    lowercase=True,
    ngram_range=(1, 2)
)

X_train_features = vectorizer.fit_transform
(X_train)
X_test_features = vectorizer.transform(X_test)

# 6. Train Model (Improved)
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

model.fit(X_train_features, y_train)

# 7. Evaluation
train_pred = model.predict(X_train_features)
test_pred = model.predict(X_test_features)

print("\nTraining Accuracy:", accuracy_score(y_train, train_pred))
print("Testing Accuracy :", accuracy_score(y_test, test_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, test_pred))

# 8. User Prediction System
print("\n========== Spam Detection System ==========")

while True:
    user_msg = input("\nEnter message (type 'exit' to quit): ")

    if user_msg.lower() == "exit":
        break

    input_features = vectorizer.transform([user_msg])
    prediction = model.predict(input_features)
    probability = model.predict_proba(input_features)

    if prediction[0] == 1:
        print("Result: Normal Mail (Ham)")
        print("Confidence:", round(probability[0][1]*100, 2), "%")
    else:
        print("Result: Spam Mail")
        print("Confidence:", round(probability[0][0]*100, 2), "%")
