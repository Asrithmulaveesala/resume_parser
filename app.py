import streamlit as st
import pickle
import re
from pypdf import PdfReader

category_mapping = {
    0: "Advocate",
    1: "Arts",
    2: "Automation Testing",
    3: "Blockchain",
    4: "Business Analyst",
    5: "Civil Engineer",
    6: "Data Science",
    7: "Database",
    8: "DevOps Engineer",
    9: "DotNet Developer",
    10: "ETL Developer",
    11: "Electrical Engineering",
    12: "HR",
    13: "Hadoop",
    14: "Health and fitness",
    15: "Java Developer",
    16: "Mechanical Engineer",
    17: "Network Security Engineer",
    18: "Operations Manager",
    19: "PMO",
    20: "Python Developer",
    21: "SAP Developer",
    22: "Sales",
    23: "Testing",
    24: "Web Designing"
}

tfidf = pickle.load(open("tfidf.pkl", "rb"))
model = pickle.load(open("knc.pkl", "rb"))

def clean_resume(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "

    return text

st.set_page_config(
    page_title="Resume Screening App",
    page_icon="📄",
    layout="centered"
)

st.title("AI Resume Screening System")
st.write("Upload your resume PDF and get the predicted job category.")

uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)

    st.subheader("Extracted Resume Preview")
    st.write(resume_text[:1000])

    if st.button("Predict Category"):
        if resume_text.strip() == "":
            st.error("Could not extract text from this PDF.")
        else:
            cleaned_resume = clean_resume(resume_text)
            vectorized_resume = tfidf.transform([cleaned_resume])

            prediction = model.predict(vectorized_resume)
            predicted_number = int(prediction[0])
            predicted_category = category_mapping.get(
                predicted_number,
                "Unknown Category"
            )

            st.success(f"Predicted Category: {predicted_category}")