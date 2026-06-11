import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Automatic Grading System")

st.title("Automatic Grading System")
st.write("Machine Learning Based Evaluation")

# Load BERT Model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Read files
with open("answer_key.txt", "r", encoding="utf-8") as f:
    answer_key = f.read()

with open("student_answer.txt", "r", encoding="utf-8") as f:
    student_answer = f.read()

# Display Answers
st.subheader("Answer Key")
st.write(answer_key)

st.subheader("Student Answer")
st.write(student_answer)

# Convert into embeddings
answer_embedding = model.encode([answer_key])
student_embedding = model.encode([student_answer])

# Similarity
similarity = cosine_similarity(
    answer_embedding,
    student_embedding
)[0][0]

score = similarity * 100
marks = round(score / 10, 2)

# Grade
if marks >= 8:
    grade = "A"
elif marks >= 6:
    grade = "B"
elif marks >= 4:
    grade = "C"
else:
    grade = "D"

# Output
st.success("Evaluation Completed")

st.metric(
    "Similarity Score",
    f"{score:.2f}%"
)

st.metric(
    "Marks",
    f"{marks}/10"
)

st.metric(
    "Grade",
    grade
)

# Save Results
data = pd.DataFrame({
    "Similarity Score": [round(score, 2)],
    "Marks": [marks],
    "Grade": [grade]
})

data.to_csv("results.csv", index=False)

st.subheader("Result")
st.dataframe(data)