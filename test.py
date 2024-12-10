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

import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import sklearn
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# %% [markdown]
# ### Ingest Data:

# %% [markdown]
# #### Question 1: Read in the data, call the dataframe "s" and check the dimensions of the dataframe

# %%
# Data Source: social media usage
s = pd.read_csv("social_media_usage.csv")

# %%
s.shape #How many rows and columns does the dataset contain?

# %% [markdown]
# ***

# %% [markdown]
# ### Examine & Clean Dataset:

# %% [markdown]
# #### Check data type of feature columns

# %%
print(s['income'].dtype)
print(s['educ2'].dtype)
print(s['age'].dtype)
print(s['par'].dtype)
print(s['marital'].dtype)
print(s['web1h'].dtype)
print(s['gender'].dtype)
print(s['sex'].dtype)

# %% [markdown]
# #### Check the total number of missing values in each column

# %%
s.isnull().sum()

# %% [markdown]
# #### Subset the dataset to keep only features specified for project

# %%
# Specify relevant columns to keep
subset_df = s[['income', 'educ2', 'age', 'par', 'marital', 'web1h', 'gender']].copy()

subset_df.head()

# %%
subset_df.shape  #dataset columns reduced from 89 to 7

# %%
#rename columns 

subset_df.rename(columns={
        'educ2': 'education',
        'par': 'parent',
        'marital': 'married',
        'web1h': 'LinkedIn_usage'
    }, inplace=True)

subset_df.head()


# %% [markdown]
# ***

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


# %%
#Create toy DataFrame to test clean_sm function
data = {'feature1': [1, 2, 1], 'feature2': [0, 1, 3]}
toy_df = pd.DataFrame(data)

print(toy_df) #view


# %%
#apply clean_sm function to toy_df

cleaned_toy_df = toy_df.map(clean_sm)

print(cleaned_toy_df) 


# %% [markdown]
# ***

# %% [markdown]
# #### Question 3: Create a new dataframe called "ss"
# 
# 

# %% [markdown]
# >The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: 
# >1. income (ordered numeric from 1 to 9, above 9 considered missing), 
# >2. education (ordered numeric from 1 to 8, above 8 considered missing), 
# >3. parent (binary), 
# >4. married (binary),
# >5. female (binary), and 
# >6. age (numeric, above 98 considered missing). 

# %%
#create dataframe ss and add binary target column
ss = pd.DataFrame(subset_df)

ss['sm_li'] = clean_sm(ss['LinkedIn_usage']).copy() #new column

ss.head()


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
# #### Perform exploratory analysis to examine how the features are related to the target -View data & Visualizatons Section for Q3

# %%
ss.head()

# %% [markdown]
# ***

# %% [markdown]
# ##### *Scatterplot indicating LinkedIn Usage by Age and Parents*

# %%
alt.Chart(ss.groupby(["age", "parent"], as_index=False)["sm_li"].mean()).\
mark_circle().\
encode(x="age",
      y="sm_li",
      color="parent:N")

# %% [markdown]
# ***

# %% [markdown]
# ##### *Plot sm_li as a function of age*

# %%
graph_df =ss.copy()

# Create age bins
graph_df['age_bins'] = pd.cut(graph_df['age'], bins=10)  

# Calculate mean LinkedIn usage by age bin
mean_usage = graph_df.groupby('age_bins')['sm_li'].mean().reset_index()

# Plot 
plt.figure(figsize=(8, 5))
sns.barplot(data=mean_usage, x='age_bins', y='sm_li', palette='viridis')
plt.title('LinkedIn Usage by Age Group', fontsize=16)
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Average LinkedIn Usage (sm_li)', fontsize=12)
plt.xticks(rotation=45, fontsize=10)  # Rotate x-axis labels for readability
plt.yticks(fontsize=10)

plt.show()


# %% [markdown]
# ***

# %% [markdown]
# ##### *Plot LinkedIn Usage by Household Income*

# %%
# Create a separate DataFrame for visualization
graph_df = ss.copy()

# Handle NaN values and convert to integers in the new DataFrame
graph_df = graph_df.dropna(subset=['income'])  # Drop rows where income is NaN
graph_df['income'] = graph_df['income'].astype(int)  # Convert to integer for mapping

# Replace numeric income categories with their descriptions
income_labels = [
    "Less than $10,000",
    "$10,000-$19,999",
    "$20,000-$29,999",
    "$30,000-$39,999",
    "$40,000-$49,999",
    "$50,000-$74,999",
    "$75,000-$99,999",
    "$100,000-$149,999",
    "$150,000 or more"
]

# Map the income labels in the new DataFrame
graph_df['income_label'] = graph_df['income'].map(lambda x: income_labels[x - 1])

# Plot LinkedIn usage by income
plt.figure(figsize=(10, 6))
sns.barplot(data=graph_df, x='income_label', y='sm_li', palette='viridis')

# Customize labels and title
plt.title('LinkedIn Usage by Household Income', fontsize=16)
plt.xlabel('Household Income', fontsize=12)
plt.ylabel('Average LinkedIn Usage (sm_li)', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

plt.show()


# %% [markdown]
# ***

# %% [markdown]
# ##### *Plot Proportion of LinkedIn Users by Gender and Marital Status*

# %%
# Create a subset with relevant columns and filter necessary rows
subset_df = ss[['gender', 'married', 'sm_li']].copy()

subset_df['female'] = subset_df['gender']  # Rename gender to female for clarity

grouped_data = subset_df.groupby(['female', 'married'])['sm_li'].mean().reset_index() # proportion of LinkedIn users female and married

grouped_data['female_label'] = grouped_data['female'].map({0: 'Male', 1: 'Female'}) 
grouped_data['married_label'] = grouped_data['married'].map({0: 'Not Married', 1: 'Married'})

grouped_data['group_label'] = grouped_data['female_label'] + ' & ' + grouped_data['married_label'] # Create a combined label for the groups

# Plot the data
plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_data, x='group_label', y='sm_li', palette='viridis')

# Customize labels and title
plt.title('Proportion of LinkedIn Usage by Gender and Marital Status', fontsize=16)
plt.xlabel('Group', fontsize=12)
plt.ylabel('Proportion of LinkedIn Usage', fontsize=12)
plt.ylim(0, 1)  # Proportion values range from 0 to 1
plt.xticks(rotation=45, fontsize=10)  # Rotate labels for clarity
plt.yticks(fontsize=10)

plt.show()


# %% [markdown]
# ***

# %% [markdown]
# ##### *Plot Proportion of Females Using LinkedIn*

# %%
# Filter for females only (gender = 1)
female_df = ss[ss['gender'] == 1]

# Calculate the proportion of LinkedIn users among females
female_linkedin_usage = female_df['sm_li'].mean()  # Proportion of sm_li = 1

# Create a DataFrame for plotting
plot_data = pd.DataFrame({
    'Category': ['Female (gender=1)'],
    'Proportion': [female_linkedin_usage]
})

# Plot
plt.figure(figsize=(6, 4))
sns.barplot(data=plot_data, x='Category', y='Proportion', palette= 'rocket')

# Customize labels and title
plt.title('Proportion of Females Using Linked')


# %% [markdown]
# ***

# %%
alt.Chart(ss.groupby(["age", "gender"], as_index=False)["sm_li"].mean()).\
mark_square().\
encode(x="age",
      y="sm_li",
      color="gender:N")

# %% [markdown]
# ***

# %% [markdown]
# ### Feature Engineering...continued

# %% [markdown]
# #### Question 4: Create a target vector (y) and feature set (X)

# %%
# Define features and target
X = ss.drop(columns=['sm_li'])  # Drop the target column to keep features
y = ss['sm_li']     


print(f"Shape of X (features): {X.shape}")
print(f"Shape of y (target): {y.shape}")

print(f"Shape of X (features): {X.info()}")


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


# %% [markdown]
# ***

# %% [markdown]
# ### Evaluate Model 

# %% [markdown]
# #### Question 7: Evaluate the model using the testing data. What is the model accuracy for the model?

# %% [markdown]
# + Use the model to make predictions and then generate a confusion matrix from the model. 
# + Interpret the confusion matrix and explain what each number means.

# %%
# Predict on the test data
y_pred = ss_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) #from sklearn.metrics import accuracy_score, confusion_matrix
print(f"Model Accuracy: {accuracy:.2f}") # Calculate and print accuracy

# Generate and print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")



# %% [markdown]
# + Model Accuracy: 1.00
# + TN (168): 168 negative samples were correctly classified as negative.
# + FP (0): 0 negative samples were incorrectly classified as positive.
# + FN (0): 0 positive samples were incorrectly classified as negative.
# + TP (84): 84 positive samples were correctly classified as positive.

# %% [markdown]
# ***

# %% [markdown]
# #### Question 8: 
# #### Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents.

# %%
pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")

# %% [markdown]
# ***

# %% [markdown]
# #### Question 9: 

# %% [markdown]
# #### Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. 
# + Use the results in the confusion matrix to calculate each of these metrics by hand.
# + Discuss each metric and give an actual example of when it might be the preferred metric of evaluation.
# + After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.

# %%
## Accuracy: TP+TN/(TP+TN+FP+FN)
# (84+168)/(84+168+0+0) = 252/252 = 1.00

## Recall: TP/(TP+FN)
# 84/(84+0) =1.00

## Precision: TP/(TP+FP)
# 84/(84+0) = 1.00

## F1 score: 2 * (Precision x Recall)/(precision + Recall) 
# 2 * [(1.00x1.00)/(1.00+1.00)] =2* [1.00/2.00] = 1.00

# %%
# Get other metrics with classification_report
print(classification_report(y_test, y_pred))

# %% [markdown]
# + Recall Example:
#   
# + Precision Example:
# 
# + F1 score Example: 
#   

# %% [markdown]
# The model has perfect performance, correctly predicting all instances in both the positive and negative classes. This is reflected in the F1 Score, precision, recall, and accuracy, all being 1.00.

# %% [markdown]
# ***

# %% [markdown]
# ### Make Predictions 

# %% [markdown]
# #### Question 10: Use the model to make predictions

# %% [markdown]
# + For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? 
# + How does the probability change if another person is 82 years old, but otherwise the same?

# %%


# %%


# %%


# %%


# %% [markdown]
# ***


