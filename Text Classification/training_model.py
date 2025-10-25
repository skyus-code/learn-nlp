import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load data
data = pd.read_csv("intent_dataset.csv")

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(data["text"], data["intent"], test_size=0.2, random_state=42)

# 3. Build pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# 4. Train model
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 6. Test
test_text = ["jam subuh di surabaya", "siapa nabi yang pertama kali"]
print(model.predict(test_text))
