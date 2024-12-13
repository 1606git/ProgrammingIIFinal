# %% [markdown]
# # Final Project- OPAN 6607 Programming II Data Infrastructure Fall 2024 
# ## Katrina Marbley
# ### December 13, 2024

# %% [markdown]
# ***

# %% [markdown]
# ### Build a classification model to predict LinkedIn Users and deploy the model on Streamlit

# %% [markdown]
# #### Import packages:

# %%
##Load all packages

import numpy as np
import matplotlib.pyplot as plt
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
#Packages to deploy app for Part 2
import streamlit as st

# %% [markdown]
# ### Ingest Data:

# %% [markdown]
# #### Question 1: Read in the data, call the dataframe "s" and check the dimensions of the dataframe

# %%
# Data Source: social media usage
s = pd.read_csv("social_media_usage.csv")

# ### Examine & Clean Dataset:

# %% [markdown]
# #### Subset the dataset to keep only features specified for project

# %%
# Specify relevant columns to keep
subset_df = s[['income', 'educ2', 'age', 'par', 'marital', 'web1h', 'gender']].copy()

#rename columns 

subset_df.rename(columns={
        'educ2': 'education',
        'par': 'parent',
        'marital': 'married'
            }, inplace=True)


# %% [markdown]
# ### Feature Engineering:

# %% [markdown]
# #### Question 2: Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1

# %% [markdown]
# + If it is, make the value of x = 1, otherwise make it 0. Return x
# + Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected

# %%
#define a function
def clean_sm(x): 
    return np.where(x == 1, 1, 0)


# %% [markdown]
# #### Question 3: Create a new dataframe called "ss"
#  
# %%
#create dataframe ss and add binary target column
ss = pd.DataFrame(subset_df).copy()

ss['sm_li'] = clean_sm(ss['web1h']).copy() #new column
ss.drop('web1h', axis=1,inplace=True)

# %%
# Process features as valid values , others set to NaN and others as binary
ss['income'] = ss['income'].apply(lambda x: x if 1 <= x <= 9 else np.nan)

ss['education'] = ss['education'].apply(lambda x: x if 1 <= x <= 8 else np.nan)

ss['parent'] = ss['parent'].apply(lambda x: 1 if x == 1 else (0))

ss['married'] = ss['married'].apply(lambda x: 1 if x == 1 else (0))

ss['gender'] = ss['gender'].apply(lambda x: 1 if x == 2 else (0))

ss['age'] = ss['age'].apply(lambda x: x if x <= 98 else np.nan)

print(ss) # Display the cleaned DataFrame


# %%
ss.isnull().sum() #check for sum of missing values

# %% [markdown]
# #### Drop any missing values in "ss"

# %%
ss = ss.dropna() #drop values considered as missing 

ss.shape #check dim of ss to validate if rows with missing values are dropped

# %% [markdown]
# ##### *Plot Proportion of Females Using LinkedIn*

# %%
# Define features and target
X = ss.drop(columns=['sm_li'])  # Drop the target column to keep features
y = ss['sm_li']     


# %% [markdown]
# ***

# %% [markdown]
# #### Question 5: Split the data into training and test sets. Hold out 20% of the data for testing

# %% [markdown]
# + Explain what each new object contains and how it is used in machine learning

# %%
# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X.values,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=987) # set for reproducibility

# Display the shapes of the resulting datasets
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# %% [markdown]
# + X_train contains 80% of the data and contains the features used to predict the target when training the model. 
# + X_test contains 20% of the data and contains the features used to test the model on unseen data to evaluate performance. 
# + y_train contains 80% of the the data and contains the target that we will predict using the features when training the model. 
# + y_test contains 20% of the data and contains the target we will predict when testing the model on unseen data to evaluate performance.
# 

# %% [markdown]
# ***

# %% [markdown]
# ### Train Logistic Regression Model

# %% [markdown]
# #### Question 6: Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.

# %%
from sklearn.linear_model import LogisticRegression

# Initialize the logistic regression model with class_weight set to 'balanced'
ss_model = LogisticRegression(class_weight='balanced')

# Fit the model with the training data
ss_model.fit(X_train, y_train)

with open("ss_model.pkl", "wb") as file:
    pickle.dump(ss_model, file)

print("Model saved successfully!")


# >Make predictions
# y_pred = ss_model.predict(X_test)
# 
# >Check accuracy
# from sklearn.metrics import accuracy_score
# #print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
# 
# >(Optional) Compare results
# #print("Predicted:", y_pred)
# #print("Actual:", y_test)
# 

# %%

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

# Helper function to preprocess inputs
def preprocess_input(income, education, age, parent, married, gender):
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

# Plotting function for prediction probabilities
def plot_probabilities(probabilities):
    plt.bar(['Not a LinkedIn User', 'LinkedIn User'], probabilities, color=['red', 'green'])
    plt.title("Prediction Confidence")
    plt.ylabel("Probability")
    st.pyplot(plt)

# Streamlit App
st.title("LinkedIn User Prediction App")

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
        prediction = km_modelmodel.predict([features])[0]
        probabilities = km_modelmodel.predict_proba([features])[0]

        st.subheader("Prediction Results")
        st.write(f"**LinkedIn User?** {'Yes' if prediction == 1 else 'No'}")
        st.write(f"**Probability of being a LinkedIn user:** {probabilities[1]:.2%}")

        st.subheader("Prediction Confidence")
        plot_probabilities(probabilities)


