
# NOTE:
# This is a starter professional Streamlit app.
# Place student_placement_dataset_100.csv in the same folder.

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Student Placement Prediction",page_icon="🎓",layout="wide")

@st.cache_resource
def train_model():
    df = pd.read_csv("student_placement_dataset_100.csv")
    X = df.drop(columns=["Student_ID","Placed"])
    y = df["Placed"]

    cat=["Primary_Skill","Certification_Level"]
    num=["CGPA","Skill_Count","Internships","Certifications",
         "Aptitude_Score","Communication_Score","Projects"]

    pre=ColumnTransformer([
        ("cat",OneHotEncoder(handle_unknown="ignore"),cat),
        ("num","passthrough",num)
    ])

    model=Pipeline([
        ("preprocessor",pre),
        ("classifier",RandomForestClassifier(n_estimators=100,random_state=42))
    ])

    model.fit(X,y)
    return model,df

model,df=train_model()

st.sidebar.title("🎓 Navigation")
page=st.sidebar.radio("Menu",["Home","Dataset Analysis","Prediction","About"])

if page=="Home":
    st.title("🎓 Student Placement Prediction Dashboard")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Students",len(df))
    c2.metric("Placed",int(df["Placed"].sum()))
    c3.metric("Average CGPA",round(df["CGPA"].mean(),2))
    c4.metric("Placement %",round(df["Placed"].mean()*100,1))

    col1,col2=st.columns(2)

    with col1:
        fig,ax=plt.subplots()
        df["Placed"].value_counts().plot(kind="pie",autopct="%1.1f%%",
                                         labels=["Placed","Not Placed"],ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

    with col2:
        fig,ax=plt.subplots()
        sns.histplot(df["CGPA"],kde=True,ax=ax)
        st.pyplot(fig)

if page=="Dataset Analysis":
    st.title("📊 Dataset Analysis")

    fig,ax=plt.subplots(figsize=(8,4))
    sns.countplot(data=df,x="Primary_Skill")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    fig,ax=plt.subplots(figsize=(8,5))
    sns.heatmap(df.corr(numeric_only=True),annot=True,cmap="coolwarm")
    st.pyplot(fig)

    fig,ax=plt.subplots(figsize=(8,4))
    sns.countplot(data=df,x="Internships",hue="Placed")
    st.pyplot(fig)

    fig,ax=plt.subplots(figsize=(8,4))
    sns.boxplot(data=df,x="Placed",y="CGPA")
    st.pyplot(fig)

if page=="Prediction":
    st.title("🤖 Placement Prediction")

    col1,col2=st.columns(2)

    with col1:
        cgpa=st.slider("CGPA",5.0,10.0,7.5)
        skill=st.selectbox("Primary Skill",
        ["Python","Java","C++","SQL","Machine Learning","Web Development",
        "Data Structures","Cloud","Power BI","Android"])
        skill_count=st.slider("Skill Count",1,10,5)
        internships=st.slider("Internships",0,5,1)

    with col2:
        certifications=st.slider("Certifications",0,10,2)
        cert_level=st.selectbox("Certification Level",
        ["None","Basic","Intermediate","Advanced"])
        aptitude=st.slider("Aptitude Score",0,100,70)
        communication=st.slider("Communication Score",0,100,70)
        projects=st.slider("Projects",0,10,3)

    if st.button("Predict"):
        sample=pd.DataFrame({
            "CGPA":[cgpa],
            "Primary_Skill":[skill],
            "Skill_Count":[skill_count],
            "Internships":[internships],
            "Certifications":[certifications],
            "Certification_Level":[cert_level],
            "Aptitude_Score":[aptitude],
            "Communication_Score":[communication],
            "Projects":[projects]
        })

        pred=model.predict(sample)[0]
        prob=max(model.predict_proba(sample)[0])*100

        if pred==1:
            st.success("🎉 Likely to be Placed")
            st.balloons()
        else:
            st.error("❌ Less Likely to be Placed")

        st.metric("Confidence",f"{prob:.2f}%")
        st.progress(prob/100)

        score=((cgpa/10)*30+(skill_count/10)*20+(internships/5)*15+
               (certifications/10)*10+(aptitude/100)*15+
               (communication/100)*10)
        st.metric("Profile Score",f"{score:.1f}/100")

        st.subheader("📌 Recommendations")
        rec=[]

        if cgpa<7.5:
            rec.append("Improve your CGPA above 7.5.")
        if internships==0:
            rec.append("Complete at least one internship.")
        if certifications<2:
            rec.append("Earn more certifications.")
        if aptitude<70:
            rec.append("Practice aptitude daily.")
        if communication<70:
            rec.append("Improve communication skills.")
        if projects<3:
            rec.append("Build more real-world projects.")
        if skill_count<4:
            rec.append("Learn more technical skills.")

        if rec:
            for r in rec:
                st.write("✅",r)
        else:
            st.success("Excellent profile! Keep preparing for interviews.")

        st.subheader("💼 Suggested Roles")
        roles={
            "Python":["Python Developer","Backend Developer","Data Analyst"],
            "Machine Learning":["ML Engineer","AI Engineer","Data Scientist"],
            "SQL":["Database Developer","Data Analyst"],
            "Web Development":["Full Stack Developer","Frontend Developer"]
        }

        for role in roles.get(skill,["Software Engineer","Graduate Trainee"]):
            st.write("•",role)

if page=="About":
    st.title("About")
    st.write("""
    ### Student Placement Prediction using Machine Learning

    **Algorithm:** Random Forest Classifier

    **Technologies**
    - Python
    - Pandas
    - Scikit-learn
    - Streamlit
    - Matplotlib
    - Seaborn
    """)
