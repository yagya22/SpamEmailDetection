import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('spam.csv', encoding='latin-1')

# Preview the first few rows
print(data.head())

# Remove unnecessary columns (adjust column names based on your dataset)
data = data[['v1', 'v2']]  # Keep only the label and text columns
data.columns = ['label', 'text']  # Rename columns for clarity

# Convert labels to binary (spam: 1, ham: 0)
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# Clean text (basic preprocessing)
data['text'] = data['text'].str.lower().str.replace(r'\W', ' ', regex=True).str.strip()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

# Confirm split
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")
