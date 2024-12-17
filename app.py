import streamlit as st
import pickle
import re

# Load saved model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Function to clean input text
def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)    # Remove special characters
    text = text.lower()                        # Convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip()   # Remove extra spaces
    return text

# Streamlit app configuration
st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f4f4f4;
    }
    .block-container {
        padding-top: 10px;
        padding-bottom: 10px;
        padding-left: 20px;
        padding-right: 20px;
        max-width: 1000px; /* Limit container width */
        margin: auto;
    }
    .stTextArea textarea {
        height: 100px !important; /* Restrict text area height */
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.title("📧 Spam Email Detector :)")
st.subheader("🔍 Enter your email content to check if it's Spam or Not Spam!")

# Add a divider
st.markdown("---")

# Text input
email_input = st.text_area(
    "✉️ Paste the email content below:",
    placeholder="Type or paste your email here...",
)

# Add a divider
st.markdown("---")

# Predict button
if st.button("🚀 Predict"):
    if email_input.strip():
        # Clean and transform input
        input_cleaned = clean_text(email_input)
        input_transformed = vectorizer.transform([input_cleaned])
        prediction = model.predict(input_transformed)[0]

        # Display results with styling
        if prediction == 1:
            st.markdown(
                """
                <div style='text-align: center; background-color: #ffdddd; padding: 20px; border-radius: 10px;'>
                <h2 style='color: red;'>🚨 Spam Detected!</h2>
                <p style='color:black;'>This email is classified as <strong>Spam</strong>.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style='text-align: center; background-color: #ddffdd; padding: 20px; border-radius: 10px;'>
                <h2 style='color: green;'>🛡️ Not Spam</h2>
                <p style='color:black;'>This email is classified as <strong>Not Spam</strong>.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.warning("⚠️ Please enter some email content to predict.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 14px;'>"
    "Created with 💖 by Yagya © 2024"
    "</div>",
    unsafe_allow_html=True,
)
