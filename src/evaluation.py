from src.model import compare_candidates

def evaluate_model(openai, data):
    correct_predictions = 0

    for index, row in data.iterrows():
        candidateA = {
            'resume': row['candidateAResume'],
            'transcript': row['candidateATranscript']
        }
        candidateB = {
            'resume': row['candidateBResume'],
            'transcript': row['candidateBTranscript']
        }
        role = row['role']
        
        prediction = compare_candidates(openai, candidateA, candidateB, role)
        
        if prediction:
            if (prediction == 'Candidate A' and row['winnerId'] == row['candidateAId']) or \
               (prediction == 'Candidate B' and row['winnerId'] == row['candidateBId']):
                correct_predictions += 1

    accuracy = correct_predictions / len(data)
    return accuracy