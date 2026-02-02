import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# 1. Vector Store & Graph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph

# 2. LLM & Embeddings
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# 3. Text Splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 4. Chains
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class GraphRAGManager:
    """
    Manages the RAG pipeline using the modern LCEL (LangChain Expression Language) architecture.
    """

    def __init__(self):
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USERNAME")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.groq_api_key = os.getenv("GROQ_API_KEY")

        # We read the model from .env, with a fallback just in case
        self.groq_model = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

        # Initialize Embeddings (Local - HuggingFace)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Initialize LLM (Groq)
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key, model_name=self.groq_model, temperature=0
        )

    def process_pdf(self, uploaded_file) -> List[str]:
        """
        Extracts text from PDF and splits it into chunks.
        """
        text = ""
        try:
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", " ", ""],
            )
            return text_splitter.split_text(text)
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return []

    def index_documents(self, text_chunks: List[str], index_name: str = "vector_index"):
        """
        Indexes the chunks into Neo4j AuraDB.
        """
        if not text_chunks:
            return None

        print("Indexing documents into Neo4j...")
        return Neo4jVector.from_texts(
            texts=text_chunks,
            embedding=self.embeddings,
            url=self.neo4j_uri,
            username=self.neo4j_user,
            password=self.neo4j_password,
            index_name=index_name,
        )

    def get_qa_chain(self):
        """
        Builds the modern RAG chain using LCEL.
        """
        # 1. Connect to existing Neo4j Index
        vector_store = Neo4jVector.from_existing_index(
            embedding=self.embeddings,
            url=self.neo4j_uri,
            username=self.neo4j_user,
            password=self.neo4j_password,
            index_name="vector_index",
        )

        # Convert vector store to a retriever
        retriever = vector_store.as_retriever()

        # 2. Define the Prompt
        system_prompt = (
            "You are a helpful assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # 3. Create the Chain
        # 'create_stuff_documents_chain' handles passing the documents to the LLM
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)

        # 'create_retrieval_chain' handles retrieving the docs + passing them to the Q&A chain
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        return rag_chain
