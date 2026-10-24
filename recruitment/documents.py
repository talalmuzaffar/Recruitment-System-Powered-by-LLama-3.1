"""Text extraction from uploaded PDF files."""
import streamlit as st
import PyPDF2
import io

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
