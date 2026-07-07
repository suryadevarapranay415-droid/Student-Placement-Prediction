import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load Dataset
# -----------------------------

@st.cache_resource
def train_model():

    df = pd.read_csv("student_placement_dataset_100.csv")

    X = df.drop(columns=["Student_ID", "Placed"])
    y = df["Placed"]

    categorical_features = [
        "Primary_Skill",
        "Certification_Level"
    ]

    numeric_features = [
        "CGPA",
        "Skill_Count",
        "Internships",
        "Certifications",
        "Aptitude_Score",
        "Communication_Score",
        "Projects"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "num",
                "passthrough",
                numeric_features
            )
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    model.fit(X, y)

    return model

model = train_model()

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Placement Prediction")

st.write(
    "Fill in the student's academic and skill details to predict placement."
)

st.divider()

# -----------------------------
# Input Fields
# -----------------------------

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
    5
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

aptitude_score = st.slider(
    "Aptitude Score",
    0,
    100,
    70
)

communication_score = st.slider(
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

st.divider()

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Placement", use_container_width=True):

    sample = pd.DataFrame({

        "CGPA":[cgpa],
        "Primary_Skill":[primary_skill],
        "Skill_Count":[skill_count],
        "Internships":[internships],
        "Certifications":[certifications],
        "Certification_Level":[certification_level],
        "Aptitude_Score":[aptitude_score],
        "Communication_Score":[communication_score],
        "Projects":[projects]

    })

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0]

    confidence = max(probability) * 100

    st.divider()

    if prediction == 1:

        st.success("🎉 Congratulations! The student is likely to be PLACED.")

    else:

        st.error("❌ The student is likely to NOT be placed.")

    st.metric(
        "Prediction Confidence",
        f"{confidence:.2f}%"
    )

    st.progress(confidence / 100)

    st.subheader("Student Details")

    st.dataframe(sample, use_container_width=True)
