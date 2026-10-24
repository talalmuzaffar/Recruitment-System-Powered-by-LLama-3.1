"""Self-RAG implementation: Groq LLM, job descriptions and the QA chain."""
import streamlit as st
from langchain.vectorstores import Qdrant
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.prompts import PromptTemplate
import json

from recruitment.config import GROQ_API_KEY
from recruitment.vector_store import init_qdrant_client, init_embedding_model

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
