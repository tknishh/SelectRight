# SelectRight

## Overview
This project aims to rank candidates for a role by comparing their resumes and interview transcripts using a language model.

## Folder Structure
```
MLE_Trial_Task/
├── data/
│   └── candidates.csv (optional, can be uploaded via the app)
├── core_services/
│   └── bot9_ai/
│       └── modules/
│           └── LLM/
│               └── OpenAi.py
├── src/
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── model.py
│   ├── evaluation.py
│   ├── bias_analysis.py
│   └── report_generation.py
├── app.py
├── requirements.txt
└── README.md
```

## Setup
1. Clone the repository.
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Files
- `data/candidates.csv`: The dataset file (optional, can be uploaded via the app).
- `llmservice/OpenAi.py`: Contains the `OpenAi` class.
- `src/data_preparation.py`: Script for loading the dataset.
- `src/model.py`: Script for defining the model.
- `src/evaluation.py`: Script for evaluating the model.
- `src/bias_analysis.py`: Script for analyzing biases.
- `src/report_generation.py`: Script for generating the report.
- `app.py`: Streamlit app script.
- `requirements.txt`: List of dependencies.
- `README.md`: Project overview and setup instructions.