"""Streamlit pages: candidate data, job descriptions and the chat interface."""
import streamlit as st
import pandas as pd

from recruitment.config import COMPANIES, JOB_TITLES
from recruitment.documents import extract_text_from_pdf
from recruitment.candidates import (
    process_candidate_data,
    get_saved_candidates,
    get_unique_values,
    delete_candidate_data,
)
from recruitment.qa import init_qa_system

# Modify the save_candidate_data function
def save_candidate_data():
    st.title("Save Candidate Data")
    
    name = st.text_input("Enter Candidate Name:")
    
    companies, job_titles = get_unique_values()
    
    company = st.selectbox("Select Company:", companies)
    job_title = st.selectbox("Select Job Title:", job_titles)
    
    uploaded_resume = st.file_uploader("Upload Resume (PDF file)", type="pdf")
    uploaded_transcript = st.file_uploader("Upload Interview Transcript (PDF file)", type="pdf")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Data"):
            if name and company and job_title and uploaded_resume and uploaded_transcript:
                try:
                    resume_text = extract_text_from_pdf(uploaded_resume)
                    interview_transcript = extract_text_from_pdf(uploaded_transcript)
                    if process_candidate_data(resume_text, interview_transcript, name, company, job_title):
                        st.success(f"Data saved successfully for {name} at {company} for the position of {job_title}")
                    else:
                        st.error("Failed to save candidate data")
                except Exception as e:
                    st.error(f"An error occurred while processing the files: {str(e)}")
            else:
                st.error("Please provide all required information and files.")
    
    with col2:
        delete_candidate = st.selectbox("Select Candidate to delete:", [""] + [name for name, _, _, _ in get_saved_candidates()])
        if delete_candidate and st.button("Delete Data"):
            delete_candidate_data(delete_candidate)
    
    st.subheader("Saved Candidates")
    saved_candidates = get_saved_candidates()
    if saved_candidates:
        df = pd.DataFrame(saved_candidates, columns=["Candidate Name", "UUID", "Company", "Job Title"])
        st.table(df)
    else:
        st.info("No candidate data saved yet.")

def chat_interface():
    st.title("Candidate Analysis Chat")
    
    # Add filtering options
    st.sidebar.title("Filter Candidates")
    companies, job_titles = get_unique_values()
    
    selected_company = st.sidebar.selectbox("Select Company:", ["All"] + companies)
    selected_job_title = st.sidebar.selectbox("Select Job Title:", ["All"] + job_titles)
    
    # Add job description selection
    st.sidebar.title("Select Job Description")
    use_jd = st.sidebar.checkbox("Use a specific Job Description")
    selected_jd_company = st.sidebar.selectbox("JD Company:", companies, disabled=not use_jd)
    selected_jd_title = st.sidebar.selectbox("JD Title:", job_titles, disabled=not use_jd)
    
    # Initialize session state variables
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome! You can now ask questions about candidates based on the selected filters."}
        ]
    if 'qa' not in st.session_state:
        st.session_state.qa = None
    
    # Initialize QA system based on filters
    if st.sidebar.button("Apply Filter and Initialize Chat"):
        with st.spinner("Initializing chat..."):
            st.session_state.messages = [
                {"role": "assistant", "content": "Chat initialized with new filters. You can now ask questions about candidates."}
            ]
            companies_to_search = [selected_company] if selected_company != "All" else COMPANIES
            st.session_state.qa = init_qa_system(companies_to_search, use_jd, selected_jd_company, selected_jd_title)
            if st.session_state.qa:
                st.success("Chat initialized with the selected filters and job description. You can now ask questions.")
            else:
                st.error("Failed to initialize chat. Please try again.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("You: "):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            if st.session_state.qa:
                # Construct a self-query string
                query = f"company: {selected_company if selected_company != 'All' else '*'} AND job_title: {selected_job_title if selected_job_title != 'All' else '*'} AND content: {prompt}"
                
                response = st.session_state.qa.run(query)
            else:
                response = "Please apply the filter and initialize the chat first."
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)

    if st.button("Start New Chat"):
        st.session_state.messages = [{"role": "assistant", "content": "Welcome! You can now ask questions about candidates based on the selected filters."}]
        st.session_state.qa = None
        st.experimental_rerun()

def save_job_description():
    st.title("Save Job Description")
    
    company = st.selectbox("Select Company:", COMPANIES)
    job_title = st.selectbox("Select Job Title:", JOB_TITLES)
    
    upload_method = st.radio("Choose input method:", ["Text Input", "File Upload"])
    
    if upload_method == "Text Input":
        job_description = st.text_area("Enter Job Description:", height=200)
    else:
        uploaded_file = st.file_uploader("Upload Job Description (TXT file)", type="txt")
        if uploaded_file is not None:
            job_description = uploaded_file.getvalue().decode("utf-8")
        else:
            job_description = ""
    
    if st.button("Save Job Description"):
        if company and job_title and job_description:
            import json
            
            job_descriptions = {}
            try:
                with open("job_descriptions.json", "r") as f:
                    job_descriptions = json.load(f)
            except FileNotFoundError:
                pass
            
            if company not in job_descriptions:
                job_descriptions[company] = {}
            job_descriptions[company][job_title] = job_description
            
            with open("job_descriptions.json", "w") as f:
                json.dump(job_descriptions, f)
            
            st.success(f"Job description saved for {job_title} at {company}")
        else:
            st.error("Please provide all required information.")
