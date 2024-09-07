# Candidate Analysis Chat App

This Streamlit app allows users to save candidate data, job descriptions, and chat with an AI assistant to analyze candidates based on their resumes and interview transcripts.

## Features

- Save candidate data (resume and interview transcript)
- Save job descriptions
- Chat interface for candidate analysis
- Filtering candidates by company and job title

## Setup

1. Clone this repository
2. Install the required packages: `pip install -r requirements.txt`
3. Set up Streamlit secrets (see below)
4. Run the app: `streamlit run app.py`

## Streamlit Secrets

When deploying to Streamlit Cloud, you need to set up the following secrets in the Streamlit Cloud dashboard:

- `QDRANT_URL`: Your Qdrant database URL
- `QDRANT_API_KEY`: Your Qdrant API key
- `GROQ_API_KEY`: Your Groq API key

For local development, create a `.streamlit/secrets.toml` file in your project directory with the following content:
