import streamlit as st
import joblib
import time

# Load the saved model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to predict fake news
def predict_news(news):
    news_tfidf = vectorizer.transform([news])
    prediction = model.predict(news_tfidf)
    return prediction[0]

# Streamlit app
st.title("Fake News Detection")

# Initialize prediction history in session state if not already initialized
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Create a sidebar for prediction history
with st.sidebar:
    st.header("Prediction History")
    history_list = st.empty()
    history_content = "\n\n".join([f"**{i+1}.** {news} - **{result}**" for i, (news, result) in enumerate(st.session_state.prediction_history)])
    history_list.markdown(history_content)

# Input text from the user
user_input = st.text_area("Enter the news text")

if st.button("Predict"):
    if user_input:
        # Display progress bar
        progress_bar = st.progress(0)
        
        # Simulate a delay for the progress bar to show
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)
        
        # Perform prediction
        prediction = predict_news(user_input)
        if prediction:
            result = "The news is Real."
        else:
            result = "The news is Fake."
        
        # Store the prediction in the session state prediction history
        st.session_state.prediction_history.append((user_input, result))
        
        # Update the sidebar with the prediction history in reverse order, but keeping numbers in ascending order
        history_content = "\n\n".join([f"**{i+1}.** {news} - **{result}**" for i, (news, result) in enumerate(st.session_state.prediction_history[::-1])])
        history_list.markdown(history_content)
        
        st.write(result)
    else:
        st.write("Please enter some news text to analyze.")
