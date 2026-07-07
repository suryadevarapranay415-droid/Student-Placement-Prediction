import streamlit as st
import pandas as pd
import joblib

# ------------------------
# Load Model
# ------------------------

model = joblib.load("placement_model.pkl")

# ------------------------
# Page Title
# ------------------------

st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Placement Prediction")

st.write("Enter the student details below.")

# ------------------------
# Inputs
# ------------------------

cgpa = st.slider(
    "CGPA",
    5.0,
    10.0,
    7.5,
    0.1
)

primary_skill = st.selectbox(
    "Primary Skill",
    [
        "Python",
        "Java",
        "C++",
        "SQL",
        "Machine Learning",
        "Web Development",
        "Data Structures",
        "Cloud",
        "Power BI",
        "Android"
    ]
)

skill_count = st.slider(
    "Number of Skills",
    1,
    10,
    4
)

internships = st.slider(
    "Internships",
    0,
    5,
    1
)

certifications = st.slider(
    "Certifications",
    0,
    10,
    2
)

certification_level = st.selectbox(
    "Certification Level",
    [
        "None",
        "Basic",
        "Intermediate",
        "Advanced"
    ]
)

aptitude = st.slider(
    "Aptitude Score",
    0,
    100,
    70
)

communication = st.slider(
    "Communication Score",
    0,
    100,
    70
)

projects = st.slider(
    "Projects Completed",
    0,
    10,
    3
)

# ------------------------
# Prediction
# ------------------------

if st.button("Predict Placement"):

    student = pd.DataFrame({

        "CGPA":[cgpa],
        "Primary_Skill":[primary_skill],
        "Skill_Count":[skill_count],
        "Internships":[internships],
        "Certifications":[certifications],
        "Certification_Level":[certification_level],
        "Aptitude_Score":[aptitude],
        "Communication_Score":[communication],
        "Projects":[projects]

    })

    prediction = model.predict(student)

    probability = model.predict_proba(student)

    confidence = probability[0][prediction[0]] * 100

    st.write("---")

    if prediction[0] == 1:

        st.success("🎉 Student is likely to get PLACED")

    else:

        st.error("❌ Student is likely to NOT get placed")

    st.subheader("Prediction Confidence")

    st.progress(int(confidence))

    st.write(f"Confidence : **{confidence:.2f}%**")

    st.write("---")

    st.subheader("Entered Details")

    st.dataframe(student)
