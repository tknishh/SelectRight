import streamlit as st
import pandas as pd
from src.data_preparation import load_data
from src.model import initialize_openai, fine_tune_model, extract_keywords, rate_skills, compare_candidates

st.title("MLE Trial Task")

# Upload OpenAI API Key
api_key = st.text_input("Enter your OpenAI API Key", type="password")
initialize_openai(api_key)

# Load training data
data = load_data("data/train_dataset.csv")

# Fine-tune the model
fine_tuned_model = fine_tune_model(data)

# Streamlit app sections
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Evaluate Test Data", "Test with Random Data"])

if page == "Evaluate Test Data":
    st.title("Evaluate Test Data")
    
    uploaded_test_file = st.file_uploader("Upload Test Data CSV", type="csv")
    
    if uploaded_test_file:
        test_data = pd.read_csv(uploaded_test_file)
        
        correct_predictions = 0
        total_predictions = len(test_data)
        
        for index, row in test_data.iterrows():
            job_description = row['role']
            candidateA_resume = row['candidateAResume']
            candidateB_resume = row['candidateBResume']
            candidateA_transcript = row['candidateATranscript']
            candidateB_transcript = row['candidateBTranscript']
            winner_id = row['winnerId']
            
            candidateA_details = {
                "keywords": extract_keywords(candidateA_resume, job_description, fine_tuned_model),
                "skills": rate_skills(candidateA_transcript, job_description, fine_tuned_model)
            }
            candidateB_details = {
                "keywords": extract_keywords(candidateB_resume, job_description, fine_tuned_model),
                "skills": rate_skills(candidateB_transcript, job_description, fine_tuned_model)
            }
            
            preferred_candidate = compare_candidates(candidateA_details, candidateB_details, job_description, fine_tuned_model)
            
            if preferred_candidate == winner_id:
                correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions
        st.write(f"Accuracy: {accuracy}")
        st.write(f"Preferred Candidates: {correct_predictions} out of {total_predictions}")

elif page == "Test with Random Data":
    st.title("Test with Random Data")
    
    job_description = st.text_area("Job Description")
    candidateA_resume = st.text_area("Candidate A Resume")
    candidateB_resume = st.text_area("Candidate B Resume")
    candidateA_transcript = st.text_area("Candidate A Transcript")
    candidateB_transcript = st.text_area("Candidate B Transcript")
    
    if st.button("Compare Candidates"):
        candidateA_details = {
            "keywords": extract_keywords(candidateA_resume, job_description, fine_tuned_model),
            "skills": rate_skills(candidateA_transcript, job_description, fine_tuned_model)
        }
        candidateB_details = {
            "keywords": extract_keywords(candidateB_resume, job_description, fine_tuned_model),
            "skills": rate_skills(candidateB_transcript, job_description, fine_tuned_model)
        }
        
        preferred_candidate = compare_candidates(candidateA_details, candidateB_details, job_description, fine_tuned_model)
        st.write(f"Preferred Candidate: {preferred_candidate}")