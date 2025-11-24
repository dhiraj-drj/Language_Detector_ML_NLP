import streamlit as st
import pandas as pd
import joblib
import os

# Configuration 
MODEL_PATH = 'NB_Language_Detector.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'
FEATURES_PATH = 'features.pkl'

# Helper Functions
@st.cache_resource
def load_resources():
    """Loads the pre-trained model, vectorizer, and expected features."""
    try:
        loaded_model = joblib.load(MODEL_PATH)
        loaded_vectorizer = joblib.load(VECTORIZER_PATH)
        expected_features = joblib.load(FEATURES_PATH)
        return loaded_model, loaded_vectorizer, expected_features
    except FileNotFoundError as e:
        st.error(f"Error loading resource: {e}. Make sure '{os.path.basename(e.filename)}' is in the correct directory.")
        st.stop() # Stop the app if resources can't be loaded
    except Exception as e:
        st.error(f"An unexpected error occurred while loading resources: {e}")
        st.stop()

def predict_language(text, model, vectorizer, expected_features):
    """
    Transforms the input text and predicts its language.
    """
    if not text:
        return "No text entered."

    # Transform the input text using the loaded vectorizer
    vectorized_input = vectorizer.transform([text])
    
    # Create a DataFrame with the vectorized input and the feature names from the vectorizer
    input_df = pd.DataFrame(vectorized_input.toarray(), columns=vectorizer.get_feature_names_out())
    
    # Align the input DataFrame with the expected features, filling missing features with 0
    # This step is crucial to ensure the input features match the model's training features
    data = pd.DataFrame(columns=expected_features).add(input_df, fill_value=0)[expected_features]
    
    # Predict the language using the loaded model
    output = model.predict(data)
    
    return output[0]

# --- Streamlit App Layout ---
def main():
    # Load resources once
    loaded_model, loaded_vectorizer, expected_features = load_resources()

    # Custom CSS for colorful background
    st.markdown(
        """
        <style>
        .stApp {
            background-image: linear-gradient(to right, #8e44ad, #3498db); /* Purple to Blue gradient */
            background-size: cover;
            background-attachment: fixed;
        }
        .stSidebar {
            background-image: linear-gradient(to bottom, #512e5f, #21618c);
            color: #ffffff; /* White text for better contrast */
        }
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff; /* White text for better contrast */
        }
        .stTextArea > label {
            color: #ffffff;
        }
        .main-title {
            color: #8B4513; /* Dark Brown color */
        }
        .stMarkdown {
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar content
    with st.sidebar:
        st.title('Language Detector 🌐')
        st.markdown("---")
        st.header("About")
        st.info(
            "This application uses a pre-trained machine learning model "
            "to detect the language of the text you enter. "
            "It's built using Naive Bayes ML Algorithm along with Streamlit, scikit-learn, and joblib. We can detect languages like Estonian, Swedish, Thai, Tamil, Dutch, Japanese, Turkish, Latin, Urdu, Indonesian, Portuguese, French, Chinese, Korean, Hindi, Spanish, Pushto, Persian, Romanian, Russian, English and Arabic."
        )
        st.markdown("---")
        st.write("Developed by Dhiraj Mandal") # Placeholder

    # Main content area
    st.markdown('<h1 class="main-title">Language Detector</h1>', unsafe_allow_html=True)
    st.header('Enter Text Below')

    # Input text area
    user_input = st.text_area('Type or paste your text here:', "", height=150)

    if st.button('Detect Language'):
        if user_input:
            predicted_language = predict_language(user_input, loaded_model, loaded_vectorizer, expected_features)
            st.success(f'Predicted Language: **{predicted_language}**')
        else:
            st.warning('Please enter some text to detect its language.')

if __name__ == '__main__':
    main()