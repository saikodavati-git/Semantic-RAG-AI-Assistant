Semantic RAG AI Assistant


A production-ready Retrieval-Augmented Generation chatbot built with Streamlit, ChromaDB, Sentence Transformers, LangChain text splitting, PyPDF, python-docx, python-dotenv, and Google Gemini 2.5 Flash.

<img width="1913" height="969" alt="Image" src="https://github.com/user-attachments/assets/c374fa5f-d397-45b4-9cb6-961b19cb41b2" />






<img width="664" height="725" alt="Image" src="https://github.com/user-attachments/assets/ac2061a4-bcbe-45cf-baeb-860b191e4b93" />



Semantic RAG AI Assistant – End-to-End System Architecture:


<img width="1096" height="724" alt="Image" src="https://github.com/user-attachments/assets/fd678e37-5247-492c-a501-452dc0be24e0" />





Features:


Multi-document upload support for PDF, DOCX, and TXT files.


Automated text extraction using PyPDF, python-docx, and TXT decoding.


Intelligent document chunking with RecursiveCharacterTextSplitter.


Duplicate-aware indexing using SHA-256 chunk identification.


Semantic embeddings powered by Sentence Transformers (all-MiniLM-L6-v2).


ChromaDB vector database for efficient document storage and retrieval.


Metadata preservation including source file, chunk index, and page numbers.


Semantic search with Top-K relevant chunk retrieval.


Context-aware question answering using Gemini 2.5 Flash.


Conversational chat experience with persistent chat history.


Transparent retrieval by displaying relevant document chunks.


Document reindexing and chat reset functionality.


Comprehensive logging for uploads, indexing, retrieval, responses, and errors.


Dockerized application for containerized deployment.


Cloud-native deployment on Google Cloud Platform (GCP) using Cloud Run



SETUP:

Create a `.env` file:


GOOGLE_API_KEY=your_google_api_key
CHROMA_DB_PATH=./vector_store
UPLOAD_DIR=./uploads



Installation:

Create and activate a virtual environment:

bash
python -m venv .venv
source .venv/bin/activate


On Windows PowerShell:

powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1



Install dependencies:

bash
pip install -r requirements.txt


Running Locally


bash
streamlit run src/app.py





Docker Deployment:


Build the Docker image:

bash
docker build -t docker image name


Run the container:

bash
docker run --env-file .env -p 8080:8080 Container_name 


Open:

http://localhost:8080




Google Cloud Run Deployment


Authenticate with Google Cloud:

bash
gcloud auth login


Set your project:

bash
gcloud config set project PROJECT_ID


Build and submit the container image:

bash
gcloud builds submit --tag gcr.io/PROJECT_ID/your docker container name


Deploy to Cloud Run:

bash
gcloud run deploy rag-chatbot \
  --image gcr.io/PROJECT_ID/dockerimagename\
  --platform managed \
  --region asia-south1 \
  --allow-unauthenticated


After deployment, configure the `GOOGLE_API_KEY` environment variable in Cloud Run. For persistent production storage, mount a supported persistent volume or use a managed vector database instead of relying only on the container filesystem.


Production Notes

Store the GOOGLE_API_KEY securely using environment variables or a secret manager.


Never upload or commit the .env file to GitHub.


Since Cloud Run storage is temporary, use a persistent storage solution or an external vector database for long-term document indexing.


Increase CPU and memory resources if faster application startup and embedding generation are required.


Duplicate detection is based on document content, preventing the same text from being indexed multiple times.


Embeddings are created only for newly added content, reducing processing time and storage costs.


Review security settings before deploying with public access (--allow-unauthenticated) to ensure only authorized users can access the application.
