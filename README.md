
# 📊 Recruitment System Powered by Llama-3.1

A powerful Streamlit app designed to streamline candidate management and analysis using advanced AI tools. Users can upload resumes and interview transcripts, save job descriptions, and chat with an AI assistant to analyze candidate profiles based on semantic search capabilities.

## ✨ Key Features

- 📝 **Candidate Data Management**: Upload and save candidate resumes and interview transcripts for easy access and analysis.
- 📑 **Job Descriptions Storage**: Save and reference job descriptions for evaluating candidates based on specific roles.
- 🤖 **AI-Powered Chat Interface**: Interact with an AI assistant to gain insights into candidate suitability using natural language queries.
- 🔍 **Filtering Options**: Filter candidates by company and job title for precise analysis.

## 🛠️ Tools and Technologies

- **Streamlit** 🖥️: Interactive user interface for candidate data management.
- **Qdrant** 🗄️: Vector database for storing and retrieving embeddings of candidate data.
- **LangChain** 🔗: Framework for integrating the AI models and managing workflows.
- **HuggingFace Embeddings** 🤗: Semantic encoding of resumes and transcripts using the `BAAI/bge-small-en` model.
- **Groq LLM** 🧠: Powerful language model for generating responses and analyzing candidates.
- **PyPDF2** 📄: Extracts text from uploaded PDF files for processing.

## 🚀 Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Streamlit secrets** (see below for details).

4. **Run the app**:
   ```bash
   streamlit run app.py
   ```

## 🔒 Streamlit Secrets Configuration

To connect to external services like Qdrant and Groq LLM, configure the following secrets:

- **QDRANT_URL** 🗄️: Your Qdrant database URL
- **QDRANT_API_KEY** 🔑: Your Qdrant API key
- **GROQ_API_KEY** 🔑: Your Groq API key

For local development, create a `.streamlit/secrets.toml` file with:

```toml
QDRANT_URL = "your_qdrant_url"
QDRANT_API_KEY = "your_qdrant_api_key"
GROQ_API_KEY = "your_groq_api_key"
```

## Project Structure

```
Recruitment-System-Powered-by-LLama-3.1/
├── app.py                # Streamlit entry point and page navigation
├── recruitment/
│   ├── __init__.py
│   ├── candidates.py     # Saving, listing and deleting candidate data
│   ├── config.py         # Streamlit secrets, companies and job titles
│   ├── documents.py      # PDF text extraction
│   ├── qa.py             # Groq LLM, job descriptions and the self-query QA chain
│   ├── ui.py             # Save Candidate Data, Save Job Description and Chat pages
│   └── vector_store.py   # Qdrant client, collections and embedding model
├── requirements.txt
└── README.md
```

## 🛠️ Example Usage

1. **Save Candidate Data**:
   - Upload resumes and interview transcripts (PDF format).
   - Enter candidate details such as name, company, and job title.

2. **Save Job Descriptions**:
   - Input job descriptions manually or upload as text files.

3. **Analyze Candidates**:
   - Use the chat interface to ask questions about candidate suitability based on the uploaded data.
   - Filter results by company and job title for focused analysis.

## 🧰 Dependencies

Make sure all dependencies are installed from `requirements.txt`:

```bash
streamlit
qdrant-client
langchain
huggingface-hub
PyPDF2
```

- **Streamlit** 🖥️: For the user interface
- **Qdrant** 🗄️: For vector database storage
- **LangChain** 🔗: To integrate AI models
- **HuggingFace Hub** 🤗: For embedding generation
- **PyPDF2** 📄: For PDF text extraction

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please open an issue or submit a pull request for improvements or new features.

## 📬 Contact

For any questions or issues, feel free to contact the project maintainers.

---

This updated `README.md` offers a clear and engaging overview of the project with helpful emojis, making it easy to follow and set up. 🎉
