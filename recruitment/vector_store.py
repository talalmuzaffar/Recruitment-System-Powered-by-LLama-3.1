"""Qdrant client, collection setup and the embedding model."""
import streamlit as st
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models

from recruitment.config import QDRANT_URL, QDRANT_API_KEY

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
