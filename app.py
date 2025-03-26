import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def download_data():
    """Download the dataset if not available."""
    url = "https://raw.githubusercontent.com/your-repo/autism_data.csv"  # Replace with actual dataset URL
    if not os.path.exists("autism_data.csv"):
        r = requests.get(url)
        with open("autism_data.csv", "wb") as f:
            f.write(r.content)

def load_data():
    """Load and preprocess data."""
    download_data()
    data = pd.read_csv("autism_data.csv")
    data.dropna(inplace=True)
    data_classes = data['Class/ASD'].apply(lambda x: 1 if x == 'YES' else 0)
    features = data.drop(columns=['Class/ASD'])
    scaler = MinMaxScaler()
    features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    return features, data_classes

def train_models(X_train, y_train, X_test, y_test):
    """Train different models and return the best one."""
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(n_estimators=5, random_state=1),
        'SVM': SVC(kernel='linear', C=1, gamma=2, probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=10),
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression()
    }
    
    best_model = None
    best_accuracy = 0
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    
    return best_model

def main():
    st.title("Autism Spectrum Disorder Prediction")
    st.write("Enter details to predict the likelihood of ASD.")
    
    # Load and Train Models
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    best_model = train_models(X_train, y_train, X_test, y_test)
    
    # User Input Fields
    user_input = {}
    for col in X.columns:
        user_input[col] = st.slider(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
    
    input_data = np.array([list(user_input.values())]).reshape(1, -1)
    
    # Predict
    if st.button("Predict"):
        prediction = best_model.predict(input_data)
        probability = best_model.predict_proba(input_data)[:, 1] if hasattr(best_model, 'predict_proba') else [0]
        
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error(f"The model predicts a high likelihood of ASD. Confidence: {probability[0]*100:.2f}%")
        else:
            st.success(f"The model predicts a low likelihood of ASD. Confidence: {(1 - probability[0])*100:.2f}%")

if __name__ == "__main__":
    main()