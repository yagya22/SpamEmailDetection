import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re

# Load the dataset
data = pd.read_csv('spam.csv', encoding='latin-1')

# Retain only necessary columns
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

# Map labels to binary values
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# Function to clean text
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove numbers and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply cleaning to the text column
data['text'] = data['text'].apply(clean_text)

# Balance the dataset
spam_data = data[data['label'] == 1]
ham_data = data[data['label'] == 0]

# Undersample ham data to balance classes
ham_data = ham_data.sample(len(spam_data), random_state=42)

# Combine balanced data
balanced_data = pd.concat([spam_data, ham_data])

# Split the balanced dataset into training and testing sets
X = balanced_data['text']
y = balanced_data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_transformed, y_train)

# Evaluate model accuracy
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test_transformed)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model and vectorizer
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
