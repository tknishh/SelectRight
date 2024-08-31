import openai
import json

def initialize_openai(api_key):
    openai.api_key = api_key

def prepare_training_data(training_data):
    training_prompts = []
    for index, row in training_data.iterrows():
        job_description = row['role']
        candidateA_resume = row['candidateAResume']
        candidateB_resume = row['candidateBResume']
        candidateA_transcript = row['candidateATranscript']
        candidateB_transcript = row['candidateBTranscript']
        winner_id = row['winnerId']
        
        prompt = f"Job Description:\n{job_description}\n\nCandidate A Resume:\n{candidateA_resume}\n\nCandidate B Resume:\n{candidateB_resume}\n\nCandidate A Transcript:\n{candidateA_transcript}\n\nCandidate B Transcript:\n{candidateB_transcript}\n\nPreferred Candidate:"
        completion = f"{winner_id}"
        
        training_prompts.append({"prompt": prompt, "completion": completion})
    
    with open("training_data.jsonl", "w") as f:
        for item in training_prompts:
            f.write(json.dumps(item) + "\n")

def upload_training_data(file_path):
    with open(file_path, "rb") as f:
        response = openai.files.create(
            file=f,
            purpose='fine-tune'
        )
    print("response-upload--->", response)
    return response.id

def create_fine_tuning_job(file_id):
    response = openai.fine_tuning.jobs.create(
        training_file=file_id,
        model="gpt-4o-2024-08-06",
    )
    print("response-create--->",response)
    return response.fine_tuned_model

def fine_tune_model(training_data):
    # Prepare training data
    prepare_training_data(training_data)
    
    # Upload training data
    file_id = upload_training_data("training_data.jsonl")
    
    # Create fine-tuning job
    fine_tuned_model = create_fine_tuning_job(file_id)
    
    return fine_tuned_model

def extract_keywords(resume, job_description, model):
    prompt = f"Extract key skills and qualifications from the following resume based on the job description:\n\nJob Description:\n{job_description}\n\nResume:\n{resume}\n\nKey Skills and Qualifications:"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    response = openai.chat.completions.create(model=model, messages=messages, max_tokens=100)
    return response.choices[0].message.content

def rate_skills(transcript, job_description, model):
    prompt = f"Rate the skills of the candidate based on the following interview transcript and job description:\n\nJob Description:\n{job_description}\n\nInterview Transcript:\n{transcript}\n\nSkill Ratings:"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    response = openai.chat.completions.create(model=model, messages=messages, max_tokens=100)
    return response.choices[0].message.content

def compare_candidates(candidateA, candidateB, job_description, model):
    prompt = f"Based on the following details, return the candidate_id of the candidate which is the best fit for the role:\n\nJob Description:\n{job_description}\n\nCandidate A:\n{candidateA}\n\nCandidate B:\n{candidateB}\n\nPreferred Candidate:, ONLY RETURN THE CANDIDATE ID which would be of format '8ab47434-09a9-44e6-8c77-f9fd20c57765'"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
        {"role": "system", "content": "<candidate_id>"}
    ]
    response = openai.chat.completions.create(model=model, messages=messages, max_tokens=100)
    return response.choices[0].message.content