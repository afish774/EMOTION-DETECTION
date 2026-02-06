import streamlit as st
from transformers import pipeline
import pandas as pd
import torch

# Set page config
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🎭",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Load the pre-trained emotion detection model."""
    return pipeline(
        "text-classification", 
        model="bhadresh-savani/distilbert-base-uncased-emotion", 
        top_k=None
    )

def main():
    st.title("🎭 Emotion Detection System")
    st.markdown("Enter a sentence below to detect the underlying emotions.")

    # Input area
    user_input = st.text_area("Your Text:", placeholder="e.g., I am so happy that I got the promotion!")

    if st.button("Analyze Emotion"):
        if not user_input.strip():
            st.warning("Please enter some text to analyze.")
            return

        with st.spinner("Analyzing..."):
            try:
                classifier = load_model()
                results = classifier(user_input)
                
                # Process results - results is a list of lists of dicts
                # [[{'label': 'joy', 'score': 0.9}, {'label': 'sadness', 'score': 0.01}, ...]]
                emotions = results[0]
                
                # Convert to DataFrame for easier plotting
                data = pd.DataFrame(emotions)
                data.columns = ["Emotion", "Confidence"]
                
                # Sort by confidence
                data = data.sort_values(by="Confidence", ascending=False)
                
                # Display dominant emotion
                top_emotion = data.iloc[0]
                st.success(f"**Dominant Emotion:** {top_emotion['Emotion'].upper()} ({top_emotion['Confidence']:.2%})")
                
                # Display chart
                st.bar_chart(data.set_index("Emotion"))
                
                # Display raw data
                with st.expander("See detailed scores"):
                    st.dataframe(data.style.format({"Confidence": "{:.2%}"}))
                    
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    main()
