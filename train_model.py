import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("student_placement_dataset_100.csv")

# Features and Target
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

# Preprocessing
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

# Pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# Train
model.fit(X, y)

# Save model
joblib.dump(model, "placement_model.pkl")

print("Model saved successfully!")
