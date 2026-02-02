# DocuNet AI: Graph-Powered AI Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Neo4j](https://img.shields.io/badge/Neo4j-AuraDB-018bff)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

> **A retrieval-augmented generation (RAG) system that combines Vector Search with Knowledge Graphs to provide accurate, context-aware answers from technical documents.**

## Overview

Traditional RAG systems often fail to capture the structural relationships between concepts. **DocuNet AI** bridges this gap by leveraging **Neo4j** to store and retrieve information not just as vectors, but as connected nodes.

This project demonstrates a production-ready implementation of a "Graph RAG" architecture using **Groq** for high-speed inference and **LangChain's modern LCEL** syntax.

### Key Features

- **Hybrid Knowledge Engine:** Integrates Vector Embeddings (HuggingFace) with Graph Database (Neo4j).
- **High-Performance LLM:** Powered by Llama 3 via Groq API for sub-second latency.
- **Modern Architecture:** Built with LangChain Expression Language (LCEL) for modularity.
- **Production Ready:** Environment configuration, error handling, and cloud-native design.

## Tech Stack

- **LLM:** Groq API (Llama-3-70b / Mixtral).
- **Graph Database:** Neo4j AuraDB (Cloud).
- **Orchestration:** LangChain (LCEL).
- **Embeddings:** `all-MiniLM-L6-v2` (Local execution via SentenceTransformers).
- **Frontend:** Streamlit.

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/gonzaloalvarezgarcia/docunet_ai.git
cd docunet_ai
```

### 2. Environment Setup

```bash
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the root directory with your credentials:

```bash
GROQ_API_KEY="your_groq_api_key"
GROQ_MODEL_NAME="llama-3.3-70b-versatile"

NEO4J_URI="neo4j+s://your-instance.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="your_neo4j_password"
```

### 4. Run the Application

```bash
streamlit run app.py
```

## Architecture

The system follows a standard ETL pipeline for RAG:

1. **Ingest**: PDF documents are uploaded and parsed.

2. **Split**: Text is divided into semantic chunks.

3. **Index**: Chunks are vectorized and stored in Neo4j.

4. **Retrieve**: User queries fetch relevant context from the Graph.

5. **Generate**: The LLM synthesizes an answer based on the retrieved context.
