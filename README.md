# Semantic Document Search Engine

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.47-red?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![Elasticsearch](https://img.shields.io/badge/Elasticsearch-8.14-005571?style=for-the-badge&logo=elasticsearch)](https://www.elastic.co/)
[![Docker](https://img.shields.io/badge/Docker-20.10-2496ED?style=for-the-badge&logo=docker)](https://www.docker.com/)
[![Hugging Face](https://img.shields.io/badge/Transformers-4.53-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/sentence-transformers)

## About The Project

This project is a powerful and intuitive semantic search engine for documents. It allows users to upload their own files (`.pdf`, `.txt`, `.csv`), which are then processed, vectorized, and indexed into an Elasticsearch database. Users can then perform natural language queries to find the most relevant text passages across the entire document base, ranked by semantic similarity.

The entire application is containerized using Docker and features a user-friendly web interface built with Streamlit.

This project demonstrates a crucial and foundational step in modern AI systems: **Semantic Search**. This technique of finding information based on meaning rather than just keywords is the core "retrieval" component in more complex architectures like **Retrieval-Augmented Generation (RAG)**. By mastering semantic search, we build the engine that feeds relevant context to Large Language Models (LLMs), enabling them to generate accurate and grounded responses.

### Features
* **File Upload:** Accepts `.pdf`, `.txt`, and `.csv` files.
* **Automatic Processing:** Extracts text, splits it into manageable chunks, and generates vector embeddings.
* **Vector Indexing:** Stores text chunks and their corresponding embeddings in Elasticsearch for efficient similarity search.
* **Semantic Search:** Allows users to query the indexed documents using natural language.
* **Interactive UI:** A clean and modern interface built with Streamlit for uploading files, managing the index, and performing searches.
* **Dockerized Environment:** Fully containerized with `docker-compose` for easy setup and deployment of the entire stack (Streamlit, Elasticsearch, Kibana).

## Tech Stack

* **Frontend:** Streamlit
* **Backend & AI Logic:** Python
* **Vector Database:** Elasticsearch (using dense_vector for KNN search)
* **Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (from Hugging Face)
* **Text Processing:** LangChain (`RecursiveCharacterTextSplitter`), PyMuPDF (for PDFs)
* **Containerization:** Docker, Docker Compose

## Getting Started

To run this project locally, you will need Docker and Docker Compose installed.

### Prerequisites

* [Docker](https://docs.docker.com/get-docker/)
* [Docker Compose](https://docs.docker.com/compose/install/) (usually included with Docker Desktop)

### Installation & Execution

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/BBennemann/semantic-search.git
    cd semantic-search
    ```

2.  **Set up environment variables:**
    Create a `.env` file in the root of the project. You can copy the structure from the `docker-compose.yml` file. A basic setup would look like this:
    ```env
    # .env
    DATA_FOLDER=data
    ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    ```

3.  **Build and run the containers:**
    From the root directory, run the following command. This will build the Streamlit app image and start all the services (Elasticsearch, Kibana, and the app).
    ```bash
    docker-compose up --build
    ```

4.  **Access the application:**
    * **Semantic Search App:** Open your browser and go to `http://localhost:8501`
    * **Kibana (Optional):** To explore the Elasticsearch index, go to `http://localhost:5601`

## How to Use the Application

1.  **Wait for the services to start:** It might take a minute for Elasticsearch to be fully operational.
2.  **Upload Documents:** Use the sidebar in the Streamlit interface to upload your `.pdf`, `.txt`, or `.csv` files.
3.  **Index Files:** Click the "Processar e Indexar Arquivos" button. The app will process the files and store them in Elasticsearch. You can see the indexed files in the "Documentos no Índice" section.
4.  **Perform a Search:** Type your query in the main search bar (e.g., "what is relativity?") and click "Buscar".
5.  **View Results:** The most relevant text chunks from your documents will be displayed, along with their source file and a similarity score.

You can also manage the index by deleting documents or re-indexing the entire database from scratch using the buttons in the sidebar.

## Contributors

* **Bernardo Thomas Bennemann** - *Project Owner* - [BBennemann](https://github.com/BBennemann)
