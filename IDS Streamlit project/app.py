import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import streamlit.components.v1 as components

# Custom CSS to style the sidebar and buttons
st.markdown("""
    <style>
    /* Change the sidebar background color to white */
    .css-1d391kg {
        background-color: black;  /* White background */
    }

    /* Style the buttons in the sidebar */
    .stButton > button {
        background-color: white;  /* Navy Blue */
        color: #000080;  /* White Text */
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
        width: 100%;  /* Full width button */
    }

    .stButton > button:hover {
        background-color: #022D36;  /* Darker Navy Blue for hover effect */
    }

    /* Change the color of the sidebar text to black */
    .css-1p7z1f8 {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Set background image

background_html = """
    <style>
    #main {
        background: rgba(255, 255, 255, 0.8);  /* White background with 80% opacity for content area */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);  /* Adding a slight shadow to make the content area stand out */
    }

    h1, h2, h3, h4, p {
        color: #003366;  /* Set text color for headings and paragraphs */
    }
    </style>
"""
# Inject custom HTML to apply the background image
components.html(background_html, height=0)

# Load the Titanic dataset from a CSV file (change the path accordingly)
df = pd.read_csv('titanic.csv')

# Custom Title and Styling
st.markdown("<h1 style='text-align: center; font-size: 40px; font-weight: bold;'>Titanic Survival Prediction Analysis</h1>", unsafe_allow_html=True)

# Add a main container for content to ensure background overlay
st.markdown('<div id="main">', unsafe_allow_html=True)

# Create a sidebar for buttons and interactions

button_overview = st.sidebar.button("Dataset Overview")
button_summary = st.sidebar.button("Summary Statistics")

# Main Visualization Section with Nested Buttons
visualization_option = st.sidebar.radio("Choose Visualization", ["None", "Visualizations"])

if visualization_option == "Visualizations":
    # Create a set of nested buttons inside this "Visualizations" category
    visualization_choice = st.sidebar.radio("Choose a Visualization", [
        "Survival Distribution (Bar Plot)",
        "Missing Values Heatmap",
        "Correlation Heatmap",
        "Age vs Survival (Scatter Plot)",
        "Survival by Class (Bar Plot)"
    ])

    # Displaying corresponding visual based on user selection
    if visualization_choice == "Survival Distribution (Bar Plot)":
        st.write("<h4 style='font-size: 20px; font-weight: bold;'>Survival Distribution</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='Survived', data=df, ax=ax, palette='Set2')
        ax.set_title("Survival Distribution", fontsize=22, fontweight='bold')
        st.pyplot(fig)

    elif visualization_choice == "Missing Values Heatmap":
        st.write("<h4 style='font-size: 20px; font-weight: bold;'>Missing Values Heatmap</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
        st.pyplot(fig)

    elif visualization_choice == "Correlation Heatmap":
        categorical_cols = df.select_dtypes(include=['object']).columns

    # Use .map() to convert categorical columns like 'yes'/'no' to numeric values
        for col in categorical_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].map({'yes': 1, 'no': 0}).fillna(df[col])

        # Perform one-hot encoding for any remaining categorical columns
        df = pd.get_dummies(df, drop_first=True)
        st.write("<h4 style='font-size: 20px; font-weight: bold;'>Correlation Heatmap</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title("Feature Correlation", fontsize=22, fontweight='bold')
        st.pyplot(fig)

    elif visualization_choice == "Age vs Survival (Scatter Plot)":
        st.write("<h4 style='font-size: 20px; font-weight: bold;'>Age vs Survival</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x='Age', y='Survived', data=df, ax=ax)
        ax.set_title("Age vs Survival", fontsize=22, fontweight='bold')
        ax.set_xlabel("Age", fontsize=18)
        ax.set_ylabel("Survival", fontsize=18)
        st.pyplot(fig)

    elif visualization_choice == "Survival by Class (Bar Plot)":
        st.write("<h4 style='font-size: 20px; font-weight: bold;'>Survival by Class</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='Pclass', hue='Survived', data=df, ax=ax, palette='Set2')
        ax.set_title("Survival by Class", fontsize=22, fontweight='bold')
        st.pyplot(fig)

# Display Dataset Overview Section
if button_overview:
    st.write("<h4 style='font-size: 20px; font-weight: bold;'>Dataset Overview</h4>", unsafe_allow_html=True)
    st.dataframe(df.head(10))  # Display first few rows of the dataset

# Display Summary Statistics Section
if button_summary:
    st.write("<h4 style='font-size: 20px; font-weight: bold;'>Summary Statistics</h4>", unsafe_allow_html=True)
    st.write(df.describe())  # Summary statistics for numerical features

# Model Evaluation Section (this section is unchanged)
button_model = st.sidebar.button("Model Evaluation")

if button_model:
    # Fill missing values in the dataset
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Convert categorical columns to numeric
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    # Prepare features and target
    X = df.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)  # Features
    y = df['Survived']  # Target

    # Scaling the features using StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(X)

    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(scaled_features, y)

    # Feature Importance
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    sorted_features = X.columns[sorted_idx]
    sorted_importance = feature_importance[sorted_idx]

    # Feature Importance Plot
    st.write("<h4 style='font-size: 20px; font-weight: bold;'>Feature Importance</h4>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=sorted_importance, y=sorted_features, ax=ax, palette='Blues')
    plt.title('Feature Importance from Random Forest Model', fontsize=22, fontweight='bold')
    st.pyplot(fig)

    # Train-Test Split Evaluation
    st.write("<h4 style='font-size: 20px; font-weight: bold;'>Train-Test Split Evaluation</h4>", unsafe_allow_html=True)

    # Train-test split, Random Forest Model Evaluation (Accuracy)
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Model Evaluation: Displaying performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'Accuracy: {accuracy:.2f}')
    st.write("<h4 style='font-size: 20px; font-weight: bold;'>Classification Report</h4>", unsafe_allow_html=True)
    st.write(classification_report(y_test, y_pred))

button_conclusion = st.sidebar.button("Conclusion")

# Conclusion Section
if button_conclusion:
    st.markdown("<h2 style='text-align: center; font-size: 32px; color: white;'>Conclusion: Key Takeaways from the Titanic Survival Prediction Project</h2>", unsafe_allow_html=True)

    # Displaying the data under key takeaways in a more structured format
    st.markdown("<h4 style='font-size: 24px; font-weight: bold; color: white;'>Data Exploration & Preprocessing:</h4>", unsafe_allow_html=True)
    st.markdown(""" 
    - The Titanic dataset provided insights into the survival patterns of passengers based on various factors like gender, age, class, and family size.
    - Preprocessing steps like handling missing values, encoding categorical variables, and feature scaling were performed to prepare the data for modeling.
    """, unsafe_allow_html=True)

    st.markdown("<h4 style='font-size: 24px; font-weight: bold; color: white;'>Modeling & Evaluation:</h4>", unsafe_allow_html=True)
    st.markdown("""
    - **Random Forest Classifier**: The Random Forest model performed well in predicting Titanic survival, leveraging important features like gender, class, and age.
    - **Evaluation Metrics**: The model demonstrated strong accuracy and was further evaluated using the classification report to assess precision, recall, and F1 score.
    """, unsafe_allow_html=True)

    st.markdown("<h4 style='font-size: 24px; font-weight: bold; color: white;'>Feature Importance:</h4>", unsafe_allow_html=True)
    st.markdown("""
    - The gender of passengers and their class had the most significant impact on survival.
    """, unsafe_allow_html=True)

# Close main container
st.markdown('</div>', unsafe_allow_html=True)
