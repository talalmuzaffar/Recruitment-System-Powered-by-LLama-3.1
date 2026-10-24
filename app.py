import streamlit as st

from recruitment import candidates
from recruitment.ui import save_candidate_data, save_job_description, chat_interface

def main():
    # SAVED_CANDIDATES lives in recruitment.candidates, so rebind it there
    candidates.SAVED_CANDIDATES = candidates.load_candidates_from_file()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Save Candidate Data", "Save Job Description", "Chat"])
    
    if page == "Save Candidate Data":
        save_candidate_data()
    elif page == "Save Job Description":
        save_job_description()
    elif page == "Chat":
        chat_interface()

if __name__ == "__main__":
    main()
