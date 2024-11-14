import streamlit as st
import joblib
from googletrans import Translator

# Load the model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
translator = Translator()

# Streamlit app
st.title("Email Spam Detection")
st.header("Welcome to the Email Spam Detection App!")

# Brief Description
st.write("""
    This application helps detect whether an email is **Spam** or **Ham** (non-spam).
    Simply paste the email content below, and the model will predict whether it's spam or not.
    
    **How it works:**
    - The model analyzes the content of the email and classifies it based on learned features.
    - If the email is not in English, the app will automatically translate it.
""")

# Input text area for email
email_text = st.text_area("Email Message", "")

# Predict and display the result
if st.button("Predict"):
    if email_text:
        # Translate if needed
        detection = translator.detect(email_text)
        if detection.lang != 'en':
            email_text = translator.translate(email_text, dest='en').text

        # Transform the text and make a prediction
        features = vectorizer.transform([email_text])
        prediction = model.predict(features)[0]
        result = 'Ham' if prediction == 1 else 'Spam'
        
        st.write(f"Prediction: **{result}**")
    else:
        st.write("Please enter some text.")

# Footer for project details
st.markdown("""
    <div style="position: fixed; bottom: 10px; right: 10px; font-size: 12px; color: gray;">
        <p>Machine Learning Semester Project by</p>
        <p>Paramveer Singh and Priyanshu Chhipa</p>
    </div>
""", unsafe_allow_html=True)
