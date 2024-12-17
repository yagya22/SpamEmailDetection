# 📧 Spam Email Detection

## 🚀 Project Overview  
The **Spam Email Detection** is a machine learning-based project that classifies emails as **Spam** or **Not Spam** using **Natural Language Processing (NLP)** and a **Logistic Regression** model. It includes a user-friendly **Streamlit** web application for real-time email classification.  

---

## ⚙️ Technologies Used  
- **Python 3.8+**  
- **Scikit-learn** - Machine learning and TF-IDF vectorization  
- **Streamlit** - Web application framework  
- **Pandas & NumPy** - Data preprocessing and numerical computations  
- **Pickle** - Model and vectorizer serialization  

---

## 📂 Project Structure  
```plaintext
spam-email-detection/
│
├── model.pkl          # Trained Logistic Regression model
├── vectorizer.pkl     # TF-IDF vectorizer
├── spam.csv           # Dataset
├── preprocess.py      # Data preprocessing and model training script
├── app.py             # Streamlit application
├── requirements.txt   # Dependencies
└── README.md          # Project documentation

```
##Clone the Repository

**git clone https://github.com/yagya22/SpamEmailDetection.git**<br/>
**cd SpamEmailDetection**

## Set Up Virtual Environment<br/>

python -m venv venv<br/>
source venv/bin/activate       # For Linux/Mac<br/>
venv\Scripts\activate          # For Windows

##Install Dependencies<br/>
**pip install -r requirement.txt**

#Run the Application<br/>
**streamlit run app.py**

