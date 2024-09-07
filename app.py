import streamlit as st
import pandas as pd
from langchain.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from uuid import uuid4
import os
import PyPDF2
import io
from langchain.prompts import PromptTemplate
import json
from collections import defaultdict

# Replace environment variables with Streamlit secrets
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ===================== SECTION 1: DATA LOADING AND MANAGEMENT =====================

# Constants
COMPANIES = ["Meta", "Google", "Amazon", "Microsoft"]
JOB_TITLES = ["Software Engineer", "Product Manager", "Data Scientist", "UX Designer"]

# Qdrant client initialization
@st.cache_resource
def init_qdrant_client(collection_name):
    try:
        client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY
        )
        create_collection_if_not_exists(client, collection_name)
        return client
    except Exception as e:
        st.error(f"Failed to initialize Qdrant client: {str(e)}")
        return None

# Embedding model initialization
@st.cache_resource
def init_embedding_model():
    try:
        model_name = "BAAI/bge-small-en"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        return HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    except Exception as e:
        st.error(f"Failed to initialize embedding model: {str(e)}")
        return None

# Function to create collection if it doesn't exist
def create_collection_if_not_exists(client, collection_name):
    try:
        collections = client.get_collections().collections
        if not any(collection.name == collection_name for collection in collections):
            vectors_config = models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
            client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
            )
    except Exception as e:
        st.error(f"Failed to create collection: {str(e)}")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Failed to extract text from PDF: {str(e)}")
        return ""

# Add this new global variable
SAVED_CANDIDATES = defaultdict(list)

# Add these new functions
def save_candidates_to_file():
    try:
        with open("saved_candidates.json", "w") as f:
            json.dump(SAVED_CANDIDATES, f)
    except Exception as e:
        st.error(f"Failed to save candidates to file: {str(e)}")

def load_candidates_from_file():
    try:
        with open("saved_candidates.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return defaultdict(list)
    except Exception as e:
        st.error(f"Failed to load candidates from file: {str(e)}")
        return defaultdict(list)

# Modify the process_candidate_data function
def process_candidate_data(resume_text, interview_transcript, name, company, job_title):
    try:
        client = init_qdrant_client(company)
        hf = init_embedding_model()
        
        if client is None or hf is None:
            return None
        
        document_store = Qdrant(
            client=client,
            collection_name=company,
            embeddings=hf
        )
        
        full_content = f"Resume:\n{resume_text}\n\nInterview Transcript:\n{interview_transcript}"
        
        document = Document(
            page_content=full_content,
            metadata={
                "name": name,
                "company": company,
                "job_title": job_title
            }
        )
        uuids = [str(uuid4())]
        document_store.add_documents(documents=[document], ids=uuids)
        
        SAVED_CANDIDATES[company].append((name, uuids[0], job_title))
        save_candidates_to_file()  # Save to file after adding a new candidate
        
        return document_store
    except Exception as e:
        st.error(f"Failed to process candidate data: {str(e)}")
        return None

# Replace the get_saved_candidates function with this
def get_saved_candidates():
    all_candidates = []
    for company, candidates in SAVED_CANDIDATES.items():
        all_candidates.extend([(name, uuid, company, job_title) for name, uuid, job_title in candidates])
    return all_candidates

def get_unique_values():
    return COMPANIES, JOB_TITLES

# Modify the delete_candidate_data function
def delete_candidate_data(candidate_name):
    try:
        saved_candidates = get_saved_candidates()
        candidate_info = next((info for info in saved_candidates if info[0] == candidate_name), None)
        if candidate_info:
            name, doc_id, company, job_title = candidate_info
            client = init_qdrant_client(company)
            if client is None:
                st.error("Failed to initialize Qdrant client")
                return
            client.delete(collection_name=company, points_selector=models.PointIdsList(points=[doc_id]))
            
            SAVED_CANDIDATES[company] = [(n, u, j) for n, u, j in SAVED_CANDIDATES[company] if n != name]
            save_candidates_to_file()  # Save to file after deleting a candidate
            
            st.success(f"Data deleted successfully for {name}")
        else:
            st.error(f"No data found for {candidate_name}")
    except Exception as e:
        st.error(f"Failed to delete candidate data: {str(e)}")

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

def filter_candidates(company=None, job_title=None):
    filtered_candidates = []
    companies_to_search = [company] if company and company != "All" else COMPANIES
    for comp in companies_to_search:
        try:
            client = init_qdrant_client(comp)
            if client is None:
                continue
            scroll_result = client.scroll(collection_name=comp, limit=100)
            documents = scroll_result[0]
            filtered_candidates.extend([
                doc.payload["name"] for doc in documents
                if (job_title is None or job_title == "All" or doc.payload.get("job_title") == job_title)
            ])
        except Exception as e:
            st.warning(f"Failed to filter candidates for {comp}: {str(e)}")
    return filtered_candidates

# ===================== SECTION 2: SELF-RAG IMPLEMENTATION =====================

# Initialize Groq LLM
@st.cache_resource
def init_groq_llm():
    try:
        return ChatGroq(
            model="llama3-groq-70b-8192-tool-use-preview",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=GROQ_API_KEY,
        )
    except Exception as e:
        st.error(f"Failed to initialize Groq LLM: {str(e)}")
        return None

# Function to initialize QA system
def init_qa_system(companies_to_search, use_jd, selected_jd_company, selected_jd_title):
    try:
        all_documents = []
        
        for company in companies_to_search:
            client = init_qdrant_client(company)
            hf = init_embedding_model()
            if client is None or hf is None:
                continue
            document_store = Qdrant(
                client=client,
                collection_name=company,
                embeddings=hf
            )
            retrieved_docs = document_store.similarity_search("", k=100)
            all_documents.extend(retrieved_docs)
        
        if not all_documents:
            st.warning("No candidates found with the selected filters.")
            return None
        
        llm = init_groq_llm()
        if llm is None:
            return None
        
        metadata_field_info = [
            AttributeInfo(
                name="name",
                description="The name of the candidate.",
                type="string",
            ),
            AttributeInfo(
                name="company",
                description="The company associated with the candidate.",
                type="string",
            ),
            AttributeInfo(
                name="job_title",
                description="The job title of the candidate.",
                type="string",
            ),
        ]
        document_content_description = "Candidate resume and interview transcript"
        
        # Load job descriptions
        job_descriptions = load_job_descriptions()
        
        # Create system prompt with job descriptions
        system_prompt = create_system_prompt(use_jd, selected_jd_company, selected_jd_title, job_descriptions)
        
        # Create a PromptTemplate
        prompt_template = PromptTemplate(
            template=system_prompt + "\n\nContext: {context}\n\nHuman: {question}\nAI: ",
            input_variables=["context", "question"]
        )

        retriever = SelfQueryRetriever.from_llm(
            llm,
            document_store,
            document_content_description,
            metadata_field_info,
            enable_limit=True
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt_template}
        )
        return qa
    except Exception as e:
        st.error(f"Failed to initialize QA system: {str(e)}")
        return None

def load_job_descriptions():
    try:
        with open("job_descriptions.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("Job descriptions file not found.")
        return {}
    except json.JSONDecodeError:
        st.error("Invalid job descriptions file format.")
        return {}
    except Exception as e:
        st.error(f"Failed to load job descriptions: {str(e)}")
        return {}

def create_system_prompt(use_jd, selected_jd_company, selected_jd_title, job_descriptions):
    system_prompt = "You are an AI assistant analyzing candidate resumes and interview transcripts. "
    if use_jd and selected_jd_company and selected_jd_title:
        job_desc = job_descriptions.get(selected_jd_company, {}).get(selected_jd_title, "")
        if job_desc:
            system_prompt += f"The job description for {selected_jd_title} at {selected_jd_company} is: {job_desc}\n"
            system_prompt += "Please provide relevant information based on the candidates' qualifications and this specific job requirement."
        else:
            st.warning(f"No job description found for {selected_jd_title} at {selected_jd_company}.")
            system_prompt += "Please provide relevant information based on the candidates' qualifications and general job requirements."
    else:
        system_prompt += "Please provide relevant information based on the candidates' qualifications and general job requirements."
    return system_prompt

# ===================== SECTION 3: CHAT INTERFACE =====================

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

def main():
    global SAVED_CANDIDATES
    SAVED_CANDIDATES = load_candidates_from_file()
    
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