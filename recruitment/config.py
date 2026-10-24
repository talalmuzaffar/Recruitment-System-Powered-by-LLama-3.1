"""Streamlit secrets and app-wide constants."""
import streamlit as st

# Replace environment variables with Streamlit secrets
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Constants
COMPANIES = ["Meta", "Google", "Amazon", "Microsoft"]
JOB_TITLES = ["Software Engineer", "Product Manager", "Data Scientist", "UX Designer"]
