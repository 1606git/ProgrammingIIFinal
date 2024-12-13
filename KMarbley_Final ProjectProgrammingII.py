# Final Project 
## Programming II
### Katrina Marbley
#### December 13, 2024

# Load Packages

import numpy as np
import pickle
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the dataset
try:
    s = pd.read_csv("social_media_usage.csv")
except FileNotFoundError:
    st.error("The file 'social_media_usage.csv' was not found. Please ensure it is in the working directory.")
    raise

# Subset and rename columns
subset_df = s[['income', 'educ2', 'age', 'par', 'marital', 'web1h', 'gender']].copy()
subset_df.rename(columns={
    'educ2': 'education',
    'par': 'parent',
    'marital': 'married'
}, inplace=True)

# Define a function to clean social media usage column
def clean_sm(x):
    """Convert to binary: 1 for LinkedIn users, 0 otherwise."""
    return (x == 1).astype(int)

# Create a cleaned DataFrame and target column
ss = subset_df.copy()
ss['sm_li'] = clean_sm(ss['web1h'])
ss.drop('web1h', axis=1, inplace=True)

# Process features with valid ranges
ss['income'] = ss['income'].apply(lambda x: x if 1 <= x <= 9 else np.nan)
ss['education'] = ss['education'].apply(lambda x: x if 1 <= x <= 8 else np.nan)
ss['parent'] = (ss['parent'] == 1).astype(int)
ss['married'] = (ss['married'] == 1).astype(int)
ss['gender'] = (ss['gender'] == 2).astype(int)  # Assuming '2' represents 'Female'
ss['age'] = ss['age'].apply(lambda x: x if x <= 98 else np.nan)

# Split data into training and test sets
X = ss.drop(columns=['sm_li'])  # Features
y = ss['sm_li']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=987)

# Train the Logistic Regression model
ss_model = LogisticRegression(class_weight='balanced')
ss_model.fit(X_train, y_train)

# Save the trained model
with open("ss_model.pkl", "wb") as file:
    pickle.dump(ss_model, file)

# Load the trained model
km_model = pickle.load(open("ss_model.pkl", "rb"))

# Define feature labels
income_labels = [
    "Less than $10,000", "$10,000-$19,999", "$20,000-$29,999",
    "$30,000-$39,999", "$40,000-$49,999", "$50,000-$74,999",
    "$75,000-$99,999", "$100,000-$149,999", "$150,000 or more"
]
education_labels = [
    "Less than high school", "High school incomplete", "High school graduate",
    "Some college, no degree", "Associate degree", "Bachelor’s degree",
    "Master’s degree", "Doctorate or professional degree"
]

# Define a function to preprocess inputs
def preprocess_input(income, education, age, parent, married, gender):
    """Process user inputs for prediction."""
    income = income if 1 <= income <= 9 else np.nan
    education = education if 1 <= education <= 8 else np.nan
    age_bin = (
        1 if age <= 18 else
        2 if age <= 35 else
        3 if age <= 55 else
        4 if age <= 75 else
        5 if age <= 98 else np.nan
    )
    return [
        income, education, age_bin,
        1 if parent == "Yes" else 0,
        1 if married == "Yes" else 0,
        1 if gender == "Female" else 0
    ]

# Streamlit App
st.title("LinkedIn User Prediction App")
st.write("Enter your details to predict LinkedIn usage and probability.")

# User inputs
income = st.selectbox("Income Range", range(1, 10), format_func=lambda x: income_labels[x - 1])
education = st.selectbox("Education Level", range(1, 9), format_func=lambda x: education_labels[x - 1])
age = st.number_input("Age (years)", min_value=0, max_value=98, step=1)
parent = st.radio("Are you a parent?", ["Yes", "No"])
married = st.radio("Are you married?", ["Yes", "No"])
gender = st.radio("Gender", ["Female", "Male"])

# Predict button
if st.button("Predict"):
    features = preprocess_input(income, education, age, parent, married, gender)

    if np.nan in features:
        st.error("Some inputs are invalid. Please check and try again.")
    else:
        prediction = km_model.predict([features])[0]
        probabilities = km_model.predict_proba([features])[0]

        # Display results
        st.subheader("Prediction Results")
        st.write(f"**LinkedIn User?** {'Yes' if prediction == 1 else 'No'}")
        st.write(f"**Probability of being a LinkedIn user:** {probabilities[1]:.2%}")

        # Prediction confidence (Bar chart)
        st.subheader("Prediction Confidence")
        st.bar_chart({
            'LinkedIn User': probabilities[1],
            'Not LinkedIn User': probabilities[0]
        })
