import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("Workplace Mental Health Support Prediction")

st.write("Enter employee details")

Age = st.number_input("Age", 18, 65)

Gender = st.selectbox("Gender", ["Male", "Female", "Other"])

Country = st.selectbox("Country", ["United States", "India", "Canada", "UK"])

self_employed = st.selectbox("Self Employed", ["Yes", "No"])

family_history = st.selectbox("Family History", ["Yes", "No"])

treatment = st.selectbox("Treatment", ["Yes", "No"])

work_interfere = st.selectbox(
    "Work Interfere",
    ["Never", "Rarely", "Sometimes", "Often"]
)

remote_work = st.selectbox("Remote Work", ["Yes", "No"])

tech_company = st.selectbox("Tech Company", ["Yes", "No"])

if st.button("Predict"):

    input_data = pd.DataFrame(
        [[Age, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        columns=[
            "Age",
            "Gender",
            "Country",
            "state",
            "self_employed",
            "family_history",
            "treatment",
            "work_interfere",
            "remote_work",
            "tech_company",
        ],
    )

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Employee will likely seek mental health support")
    else:
        st.warning("Employee may not seek mental health support")