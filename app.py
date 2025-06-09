import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
df = pd.read_csv("diabetes.csv")

# Title
st.title("Diabetes Prediction using SVM")

# Input form for user input
st.header("Enter Patient Data")
with st.form("input_form"):
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 200, 120)
    blood_pressure = st.number_input("Blood Pressure", 0, 140, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.number_input("Age", 10, 100, 33)
    submit_button = st.form_submit_button(label="Predict")

# Collect input into DataFrame
user_data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [dpf],
    'Age': [age]
})

# Prepare data for training
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train SVM model
model = SVC()
model.fit(X_train, y_train)

# Prediction on test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Prediction button
if submit_button:
    user_input_scaled = scaler.transform(user_data)
    user_prediction = model.predict(user_input_scaled)

    st.subheader("Model Test Accuracy")
    st.write(f"{accuracy * 100:.2f}%")

    st.subheader("Prediction")
    pred_label = 'Diabetic' if user_prediction[0] == 1 else 'Non-Diabetic'
    st.write(f"The model predicts: **{pred_label}**")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Diabetic', 'Diabetic'])
    disp.plot(ax=ax)
    st.pyplot(fig)

   
# Additional Graphs
st.subheader("Data Distribution")
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(data=df, x="Age", hue="Outcome", bins=30, kde=True, ax=ax)
st.pyplot(fig)

fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x="Outcome", y="Glucose", ax=ax2)
st.pyplot(fig2)
