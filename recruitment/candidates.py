"""Saving, listing and deleting candidate data."""
import streamlit as st
from langchain.vectorstores import Qdrant
from qdrant_client.http import models
from langchain_core.documents import Document
from uuid import uuid4
import json
from collections import defaultdict

from recruitment.config import COMPANIES, JOB_TITLES
from recruitment.vector_store import init_qdrant_client, init_embedding_model

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
