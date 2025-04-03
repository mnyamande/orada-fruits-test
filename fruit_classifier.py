"""
Fruit Classifier Exercise
ML Workshop Template

This file contains a complete Streamlit application for the fruit classifier exercise.
Participants can use this template to:
1. Train a text-based fruit classifier
2. Save and load the model
3. Deploy the model as a web application
4. Test the model with new input data
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Create necessary directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="Fruit Classifier",
    page_icon="🍎",
    layout="wide"
)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = 0.0

# Default dataset - participants can expand this
default_data = {
    'features': [
        'round red sweet crisp grows on trees',
        'yellow curved sweet soft tropical',
        'small round orange citrus juicy',
        'green oval soft large seed tropical',
        'small round purple sweet grows on vines',
        'yellow round sour citrus',
        'red small sweet juicy grows on bushes',
        'green round hard tropical hairy brown inside',
        'orange oval sweet soft',
        'red oblong juicy sweet'
    ],
    'fruit': [
        'apple',
        'banana',
        'orange',
        'avocado',
        'grape',
        'lemon',
        'strawberry',
        'kiwi',
        'papaya',
        'watermelon'
    ]
}

# Functions for model training and prediction
def train_model(data_df, vectorizer_type='count'):
    """Train a fruit classifier model using text features"""
    # Create vectorizer
    if vectorizer_type == 'count':
        vectorizer = CountVectorizer()
    else:
        vectorizer = TfidfVectorizer()
    
    # Transform text features
    X = vectorizer.fit_transform(data_df['features'])
    y = data_df['fruit']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.01, random_state=42
    )
    
    # Train model
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save objects in session state
    st.session_state.vectorizer = vectorizer
    st.session_state.classifier = classifier
    st.session_state.model_trained = True
    st.session_state.accuracy = accuracy
    
    # Save model files
    with open('models/fruit_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('models/fruit_classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    
    return vectorizer, classifier, accuracy

def predict_fruit(features, vectorizer, classifier):
    """Make a prediction using the trained model"""
    # Transform input text
    X_input = vectorizer.transform([features])
    
    # Make prediction
    prediction = classifier.predict(X_input)[0]
    
    # Get probabilities
    probabilities = classifier.predict_proba(X_input)[0]
    
    # Get fruit names and their probabilities
    fruit_probs = {
        fruit: prob 
        for fruit, prob in zip(classifier.classes_, probabilities)
    }
    
    # Sort by probability (descending)
    sorted_probs = dict(sorted(
        fruit_probs.items(), 
        key=lambda x: x[1], 
        reverse=True
    ))
    
    return prediction, sorted_probs

def load_model():
    """Load a previously saved model"""
    try:
        with open('models/fruit_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('models/fruit_classifier.pkl', 'rb') as f:
            classifier = pickle.load(f)
        
        st.session_state.vectorizer = vectorizer
        st.session_state.classifier = classifier
        st.session_state.model_trained = True
        
        return vectorizer, classifier
    except FileNotFoundError:
        st.error("Model files not found. Please train your model first.")
        return None, None

# Main application layout
def main():
    st.title("🍎 Fruit Classifier")
    st.markdown("""
    This application demonstrates a simple text-based fruit classifier.
    Enter descriptions of fruits, and the model will predict which fruit it is.
    """)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs([
        "Train Model", 
        "Test Model", 
        "Advanced Options"
    ])
    
    # Tab 1: Train Model
    with tab1:
        st.header("Train Your Fruit Classifier")
        
        # Dataset options
        dataset_option = st.radio(
            "Choose dataset option:",
            ["Use default dataset", "Upload your own dataset"]
        )
        
        if dataset_option == "Use default dataset":
            data_df = pd.DataFrame(default_data)
            st.dataframe(data_df)
        else:
            st.write("Upload a CSV file with 'features' and 'fruit' columns:")
            uploaded_file = st.file_uploader(
                "Choose a CSV file", 
                type="csv"
            )
            
            if uploaded_file is not None:
                try:
                    data_df = pd.read_csv(uploaded_file)
                    if 'features' in data_df.columns and 'fruit' in data_df.columns:
                        st.dataframe(data_df)
                    else:
                        st.error("CSV must contain 'features' and 'fruit' columns")
                        data_df = pd.DataFrame(default_data)
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
                    data_df = pd.DataFrame(default_data)
            else:
                data_df = pd.DataFrame(default_data)
        
        # Training options
        st.subheader("Training Options")
        vectorizer_type = st.selectbox(
            "Choose vectorizer type:",
            ["count", "tfidf"]
        )
        
        # Train button
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                try:
                    vectorizer, classifier, accuracy = train_model(
                        data_df, 
                        vectorizer_type
                    )
                    st.success(f"Model trained successfully! Accuracy: {accuracy:.2f}")
                    
                    # Show model details
                    st.subheader("Model Details")
                    st.write(f"Number of features: {len(vectorizer.get_feature_names_out())}")
                    st.write(f"Top features: {', '.join(vectorizer.get_feature_names_out()[:10])}")
                    st.write(f"Number of fruit classes: {len(classifier.classes_)}")
                    st.write(f"Fruit classes: {', '.join(classifier.classes_)}")
                except Exception as e:
                    st.error(f"Error training model: {e}")
        
        # Load model button
        if st.button("Load Saved Model"):
            with st.spinner("Loading model..."):
                vectorizer, classifier = load_model()
                if vectorizer is not None and classifier is not None:
                    st.success("Model loaded successfully!")
    
    # Tab 2: Test Model
    with tab2:
        st.header("Test Your Fruit Classifier")
        
        # Check if model is trained
        if not st.session_state.model_trained:
            st.warning("Please train or load a model first.")
        else:
            # Input features
            features = st.text_area(
                "Enter fruit description:",
                "round red sweet",
                help="Describe the fruit's shape, color, taste, and other characteristics."
            )
            
            # Predict button
            if st.button("Predict Fruit"):
                with st.spinner("Making prediction..."):
                    prediction, probabilities = predict_fruit(
                        features, 
                        st.session_state.vectorizer, 
                        st.session_state.classifier
                    )
                    
                    # Show prediction
                    st.subheader("Prediction")
                    st.success(f"This is most likely a: **{prediction}**")
                    
                    # Show probabilities
                    st.subheader("Confidence Levels")
                    
                    # Create a DataFrame for visualization
                    probs_df = pd.DataFrame({
                        'Fruit': list(probabilities.keys()),
                        'Probability': list(probabilities.values())
                    })
                    
                    # Only show top 5 predictions
                    top_probs = probs_df.head(5)
                    
                    # Display as bar chart
                    st.bar_chart(
                        top_probs.set_index('Fruit')
                    )
                    
                    # Display as table
                    st.dataframe(
                        top_probs.style.format({'Probability': '{:.2%}'})
                    )
    
    # Tab 3: Advanced Options
    with tab3:
        st.header("Advanced Options")
        
        # Export model
        st.subheader("Export Model")
        if st.session_state.model_trained:
            st.download_button(
                label="Download Vectorizer",
                data=open('models/fruit_vectorizer.pkl', 'rb'),
                file_name='fruit_vectorizer.pkl',
                mime='application/octet-stream'
            )
            
            st.download_button(
                label="Download Classifier",
                data=open('models/fruit_classifier.pkl', 'rb'),
                file_name='fruit_classifier.pkl',
                mime='application/octet-stream'
            )
        else:
            st.warning("Please train a model first.")
        
        # Model inspection
        st.subheader("Model Inspection")
        if st.session_state.model_trained:
            st.write("Vectorizer vocabulary size:", 
                     len(st.session_state.vectorizer.get_feature_names_out()))
            
            # Show top features for each class
            st.write("Top features for each class:")
            
            # Get feature names
            feature_names = st.session_state.vectorizer.get_feature_names_out()
            
            # Iterate through classes and show top features
            for i, fruit_class in enumerate(st.session_state.classifier.classes_):
                # Get feature importance for this class
                feature_importance = st.session_state.classifier.feature_log_prob_[i]
                
                # Sort features by importance
                sorted_indices = np.argsort(feature_importance)[::-1]
                
                # Get top 5 features
                top_features = [feature_names[idx] for idx in sorted_indices[:5]]
                
                st.write(f"**{fruit_class}**: {', '.join(top_features)}")
        else:
            st.warning("Please train a model first.")

# Run the application
if __name__ == "__main__":
    main()
